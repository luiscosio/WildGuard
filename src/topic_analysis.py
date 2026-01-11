"""Topic clustering analysis - Do certain topics show more dark patterns?"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Any
import numpy as np
import openai
from loguru import logger
from scipy import stats

from .config import OUTPUTS_DIR, LLMConfig
from .utils import load_jsonl, save_jsonl, save_json


def extract_first_user_messages(wildchat_file: Path) -> dict[str, str]:
    """Extract the first user message from each conversation."""
    # Load the raw WildChat data to get user messages
    logger.info(f"Loading WildChat data from {wildchat_file}")

    conversations = defaultdict(list)

    # We need the original WildChat data with user messages
    # Let's load and parse
    data = load_jsonl(wildchat_file)

    for item in data:
        conv_id = item.get("conversation_id", "")
        if conv_id:
            conversations[conv_id].append(item)

    # Get first user message per conversation (if we have it)
    # Note: Our turns file only has assistant messages, so we need original data
    # For now, let's use the conversation context from detections

    logger.info(f"Found {len(conversations)} conversations")
    return conversations


def cluster_conversations(
    detections_file: Path,
    wildchat_file: Path,
    n_clusters: int = 10,
    sample_per_cluster: int = 5,
) -> tuple[dict, np.ndarray, list]:
    """Cluster conversations by topic using embeddings."""
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    logger.info("Loading detections...")
    detections = load_jsonl(detections_file)

    # Group by conversation, get first turn content as proxy for topic
    conv_first_turns = {}
    conv_flags = defaultdict(list)

    for det in detections:
        conv_id = det.get("conversation_id", "")
        turn_idx = det.get("turn_index", 0)
        content = det.get("content", "")
        category = det.get("predicted_category", "none")

        # Store first turn (assistant response to first user query)
        if conv_id not in conv_first_turns or turn_idx < conv_first_turns[conv_id]["turn_index"]:
            conv_first_turns[conv_id] = {
                "turn_index": turn_idx,
                "content": content[:1000],  # Truncate for embedding
                "conv_id": conv_id,
            }

        # Track all flags
        conv_flags[conv_id].append(category)

    logger.info(f"Found {len(conv_first_turns)} conversations with first turns")

    # Prepare for clustering
    conv_ids = list(conv_first_turns.keys())
    texts = [conv_first_turns[cid]["content"] for cid in conv_ids]

    # Embed
    logger.info("Embedding first turns...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True)

    # Cluster
    logger.info(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Analyze each cluster
    cluster_data = {}
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_conv_ids = [conv_ids[i] for i in range(len(conv_ids)) if mask[i]]

        # Get sample texts for labeling
        sample_indices = random.sample(range(len(cluster_conv_ids)),
                                        min(sample_per_cluster, len(cluster_conv_ids)))
        samples = [conv_first_turns[cluster_conv_ids[i]]["content"][:500] for i in sample_indices]

        # Compute stats
        total_turns = 0
        flagged_turns = 0
        category_counts = defaultdict(int)

        for cid in cluster_conv_ids:
            flags = conv_flags[cid]
            total_turns += len(flags)
            for cat in flags:
                if cat != "none":
                    flagged_turns += 1
                    category_counts[cat] += 1

        flag_rate = flagged_turns / total_turns if total_turns > 0 else 0

        cluster_data[cluster_id] = {
            "n_conversations": len(cluster_conv_ids),
            "total_turns": total_turns,
            "flagged_turns": flagged_turns,
            "flag_rate": flag_rate,
            "category_counts": dict(category_counts),
            "sample_texts": samples,
            "conv_ids": cluster_conv_ids[:100],  # Store first 100 for reference
        }

    return cluster_data, labels, conv_ids


def call_llm(prompt: str, config: LLMConfig, max_retries: int = 3) -> str:
    """Call LLM via OpenRouter with retry logic."""
    import time
    client = openai.OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 2}/{max_retries}")
                time.sleep(wait_time)
            else:
                raise
    return ""


def label_clusters_with_llm(
    cluster_data: dict,
    config: LLMConfig,
) -> dict:
    """Use LLM to label each cluster with a topic name."""
    import time
    logger.info("Labeling clusters with LLM...")

    for cluster_id, data in cluster_data.items():
        time.sleep(2)  # Rate limit protection
        samples = data["sample_texts"]
        samples_text = "\n---\n".join(f"Sample {i+1}:\n{s}" for i, s in enumerate(samples))

        prompt = f"""Analyze these sample AI assistant responses from the same topic cluster.
What is the common topic or user intent? Provide a short label (2-4 words).

{samples_text}

Respond with ONLY the topic label, nothing else. Examples: "Coding Help", "Creative Writing", "Emotional Support", "Factual Q&A", "Language Translation", "Roleplay/Fiction"."""

        try:
            response = call_llm(prompt, config)
            label = response.strip().strip('"').strip("'")
            # Truncate if too long
            if len(label) > 30:
                label = label[:30]
            data["topic_label"] = label
            logger.info(f"Cluster {cluster_id}: {label} ({data['n_conversations']} convs, {data['flag_rate']*100:.1f}% flags)")
        except Exception as e:
            logger.warning(f"Failed to label cluster {cluster_id}: {e}")
            data["topic_label"] = f"Cluster {cluster_id}"

    return cluster_data


def compute_escalation_per_cluster(
    detections_file: Path,
    cluster_assignments: dict[str, int],
) -> dict:
    """Compute escalation slope per cluster."""
    logger.info("Computing per-cluster escalation...")

    detections = load_jsonl(detections_file)

    # Group by cluster and turn index
    cluster_turn_data = defaultdict(lambda: defaultdict(list))

    for det in detections:
        conv_id = det.get("conversation_id", "")
        turn_idx = det.get("turn_index", 0)
        category = det.get("predicted_category", "none")

        if conv_id in cluster_assignments:
            cluster_id = cluster_assignments[conv_id]
            is_flagged = 1 if category != "none" else 0
            cluster_turn_data[cluster_id][turn_idx].append(is_flagged)

    # Compute regression per cluster
    escalation_results = {}
    for cluster_id, turn_data in cluster_turn_data.items():
        turns = []
        flags = []
        for turn_idx, flag_list in turn_data.items():
            for f in flag_list:
                turns.append(turn_idx)
                flags.append(f)

        if len(turns) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(turns, flags)
            escalation_results[cluster_id] = {
                "slope": float(slope),
                "p_value": float(p_value),
                "r_squared": float(r_value ** 2),
                "significant": bool(p_value < 0.05),
            }
        else:
            escalation_results[cluster_id] = None

    return escalation_results


def run_topic_analysis(
    detections_file: Path,
    wildchat_file: Path,
    output_file: Path,
    n_clusters: int = 10,
    config: LLMConfig = None,
) -> dict:
    """Run full topic analysis pipeline."""

    # Cluster
    cluster_data, labels, conv_ids = cluster_conversations(
        detections_file, wildchat_file, n_clusters
    )

    # Label with LLM
    if config:
        cluster_data = label_clusters_with_llm(cluster_data, config)

    # Create conv_id -> cluster mapping
    cluster_assignments = {conv_ids[i]: labels[i] for i in range(len(conv_ids))}

    # Compute escalation
    escalation = compute_escalation_per_cluster(detections_file, cluster_assignments)

    # Add escalation to cluster data
    for cluster_id in cluster_data:
        cluster_data[cluster_id]["escalation"] = escalation.get(cluster_id)

    # Summary table
    results = {
        "n_clusters": n_clusters,
        "total_conversations": len(conv_ids),
        "clusters": {},
    }

    for cluster_id, data in sorted(cluster_data.items(), key=lambda x: -x[1]["flag_rate"]):
        results["clusters"][cluster_id] = {
            "topic_label": data.get("topic_label", f"Cluster {cluster_id}"),
            "n_conversations": data["n_conversations"],
            "total_turns": data["total_turns"],
            "flagged_turns": data["flagged_turns"],
            "flag_rate": data["flag_rate"],
            "category_counts": data["category_counts"],
            "escalation": data.get("escalation"),
        }

    # Save
    save_json(results, output_file)
    logger.info(f"Topic analysis saved to {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("TOPIC ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'Topic':<25} {'Convs':>8} {'Flag Rate':>10} {'Escalation':>12} {'p-value':>10}")
    print("-"*80)
    for cid, cdata in sorted(results["clusters"].items(), key=lambda x: -x[1]["flag_rate"]):
        label = cdata["topic_label"][:24]
        esc = cdata.get("escalation")
        esc_str = f"{esc['slope']*1000:.2f}/1k" if esc else "N/A"
        p_str = f"{esc['p_value']:.2e}" if esc and esc['p_value'] else "N/A"
        print(f"{label:<25} {cdata['n_conversations']:>8} {cdata['flag_rate']*100:>9.2f}% {esc_str:>12} {p_str:>10}")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Topic clustering analysis")
    parser.add_argument(
        "--detections",
        type=str,
        default=str(OUTPUTS_DIR / "v5" / "wildchat_detections_v5b.jsonl"),
        help="Classifier detections file",
    )
    parser.add_argument(
        "--wildchat",
        type=str,
        default=str(OUTPUTS_DIR / "v2" / "wildchat_turns_100k.jsonl"),
        help="WildChat turns file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "topic_analysis.json"),
        help="Output file",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of topic clusters",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM labeling",
    )
    args = parser.parse_args()

    config = None
    if not args.no_llm:
        config = LLMConfig(
            provider="openrouter",
            model="anthropic/claude-3-5-haiku-20241022",
        )

    run_topic_analysis(
        Path(args.detections),
        Path(args.wildchat),
        Path(args.output),
        n_clusters=args.n_clusters,
        config=config,
    )


if __name__ == "__main__":
    main()
