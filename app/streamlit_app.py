"""WildGuard Streamlit Demo App - Dark Pattern Monitoring Dashboard."""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUTS_DIR, FIGURES_DIR, DARK_PATTERN_CATEGORIES
from src.utils import load_jsonl, load_json
from src.infer_wildchat import DarkPatternClassifier

# Page configuration
st.set_page_config(
    page_title="WildGuard - Dark Pattern Monitor",
    page_icon="🛡️",
    layout="wide",
)


def load_sample_data():
    """Load sample detection data for demo."""
    # Try V5 path first
    detections_path = OUTPUTS_DIR / "v5" / "wildchat_detections_v5b.jsonl"
    if not detections_path.exists():
        detections_path = OUTPUTS_DIR / "wildchat_detections.jsonl"  # Fallback
    if detections_path.exists():
        return load_jsonl(detections_path)

    # Generate sample data if not exists
    return [
        {
            "id": f"sample_{i}",
            "content": f"Sample assistant response {i}",
            "predicted_category": DARK_PATTERN_CATEGORIES[i % len(DARK_PATTERN_CATEGORIES)],
            "predicted_confidence": 0.5 + (i % 5) * 0.1,
            "turn_index": i % 10,
        }
        for i in range(100)
    ]


def load_reports():
    """Load analytics reports."""
    reports = {}

    # Try V5 paths first, fallback to root outputs
    analytics_path = OUTPUTS_DIR / "v5" / "analytics_v5b.json"
    if not analytics_path.exists():
        analytics_path = OUTPUTS_DIR / "prevalence.json"
    if analytics_path.exists():
        reports["prevalence"] = load_json(analytics_path)

    gap_path = OUTPUTS_DIR / "v5" / "gap_report.json"
    if not gap_path.exists():
        gap_path = OUTPUTS_DIR / "gap_report.json"
    if gap_path.exists():
        reports["gap"] = load_json(gap_path)

    reliability_path = OUTPUTS_DIR / "v5" / "reliability_report.json"
    if not reliability_path.exists():
        reliability_path = OUTPUTS_DIR / "reliability_report.json"
    if reliability_path.exists():
        reports["reliability"] = load_json(reliability_path)

    # Topic analysis
    topic_path = OUTPUTS_DIR / "v5" / "topic_analysis.json"
    if topic_path.exists():
        reports["topic"] = load_json(topic_path)

    return reports


def render_sidebar():
    """Render sidebar navigation."""
    st.sidebar.title("🛡️ WildGuard")
    st.sidebar.markdown("Dark Pattern Monitoring System")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Scan Transcript", "📈 Analytics", "📋 Reports"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "WildGuard monitors LLM conversations for dark patterns "
        "using a combination of classifier and LLM judge approaches."
    )

    return page


def render_dashboard():
    """Render main dashboard."""
    st.title("📊 Dark Pattern Monitoring Dashboard")

    reports = load_reports()
    detections = load_sample_data()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    total_scanned = len(detections)
    flagged = [d for d in detections if d.get("predicted_category", "none") != "none"]

    with col1:
        st.metric("Total Turns Scanned", f"{total_scanned:,}")

    with col2:
        st.metric("Flagged Turns", f"{len(flagged):,}")

    with col3:
        flag_rate = len(flagged) / total_scanned if total_scanned > 0 else 0
        st.metric("Flag Rate", f"{flag_rate:.1%}")

    with col4:
        if "reliability" in reports:
            score = reports["reliability"].get("summary", {}).get("reliability_score", 0)
            st.metric("Reliability Score", f"{score:.0%}")
        else:
            st.metric("Reliability Score", "N/A")

    st.markdown("---")

    # Category distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category Distribution")

        category_counts = {}
        for d in detections:
            cat = d.get("predicted_category", "none")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        df_cats = pd.DataFrame([
            {"Category": cat, "Count": count}
            for cat, count in category_counts.items()
            if cat != "none"
        ])

        if not df_cats.empty:
            st.bar_chart(df_cats.set_index("Category"))
        else:
            st.info("No category data available")

    with col2:
        st.subheader("Confidence Distribution")

        confidences = [d.get("predicted_confidence", 0) for d in flagged]
        if confidences:
            df_conf = pd.DataFrame({"Confidence": confidences})
            st.bar_chart(df_conf["Confidence"].value_counts().sort_index())
        else:
            st.info("No confidence data available")

    # Recent detections
    st.subheader("Recent High-Risk Detections")

    high_risk = sorted(
        [d for d in flagged if d.get("predicted_confidence", 0) > 0.6],
        key=lambda x: -x.get("predicted_confidence", 0),
    )[:10]

    if high_risk:
        for det in high_risk:
            with st.expander(
                f"**{det.get('predicted_category', 'unknown')}** "
                f"(confidence: {det.get('predicted_confidence', 0):.0%})"
            ):
                st.markdown(f"**Content:**")
                st.text(det.get("content", "N/A")[:500])
                st.markdown(f"**Turn Index:** {det.get('turn_index', 'N/A')}")
    else:
        st.info("No high-risk detections to display")


def render_scan_page():
    """Render transcript scanning page."""
    st.title("🔍 Scan Transcript")

    st.markdown(
        "Upload a chat transcript (JSONL format) or paste text to scan for dark patterns."
    )

    tab1, tab2 = st.tabs(["📄 Upload File", "✏️ Paste Text"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload JSONL transcript",
            type=["jsonl", "json"],
        )

        if uploaded_file:
            try:
                content = uploaded_file.read().decode("utf-8")
                lines = content.strip().split("\n")
                records = [json.loads(line) for line in lines if line.strip()]

                st.success(f"Loaded {len(records)} records")

                if st.button("Scan for Dark Patterns"):
                    with st.spinner("Analyzing..."):
                        # Initialize classifier
                        from src.config import MODELS_DIR
                        classifier = DarkPatternClassifier(
                            model_dir=MODELS_DIR / "classifier",
                            model_type="embedding",
                        )

                        # Extract texts
                        texts = [r.get("content", r.get("text", "")) for r in records]
                        predictions = classifier.predict(texts)

                        # Display results
                        st.subheader("Scan Results")

                        flagged = [
                            (r, p) for r, p in zip(records, predictions)
                            if p["category"] != "none"
                        ]

                        st.metric("Flagged Items", len(flagged))

                        for record, pred in flagged:
                            with st.expander(
                                f"**{pred['category']}** ({pred['confidence']:.0%})"
                            ):
                                st.text(record.get("content", record.get("text", ""))[:500])

            except Exception as e:
                st.error(f"Error processing file: {e}")

    with tab2:
        text_input = st.text_area(
            "Paste assistant response to analyze",
            height=200,
            placeholder="Enter the assistant's response here...",
        )

        if st.button("Analyze Text"):
            if text_input:
                with st.spinner("Analyzing..."):
                    from src.config import MODELS_DIR
                    classifier = DarkPatternClassifier(
                        model_dir=MODELS_DIR / "classifier",
                        model_type="embedding",
                    )

                    predictions = classifier.predict([text_input])
                    pred = predictions[0]

                    st.subheader("Analysis Result")

                    if pred["category"] != "none":
                        st.warning(
                            f"⚠️ Detected: **{pred['category']}** "
                            f"(confidence: {pred['confidence']:.0%})"
                        )
                    else:
                        st.success("✅ No dark pattern detected")

                    # Show probabilities
                    st.subheader("Category Probabilities")
                    probs = pred.get("class_probabilities", {})
                    df_probs = pd.DataFrame([
                        {"Category": k, "Probability": v}
                        for k, v in probs.items()
                    ]).sort_values("Probability", ascending=False)
                    st.bar_chart(df_probs.set_index("Category"))
            else:
                st.warning("Please enter some text to analyze")


def render_analytics_page():
    """Render analytics page."""
    st.title("📈 Analytics")

    reports = load_reports()
    detections = load_sample_data()

    # Prevalence section
    st.subheader("Prevalence by Category")

    if "prevalence" in reports:
        prev_data = reports["prevalence"].get("prevalence", {}).get("prevalence_by_category", {})
        if prev_data:
            df_prev = pd.DataFrame([
                {"Category": cat, "Per 1000 Turns": data.get("per_1000", 0)}
                for cat, data in prev_data.items()
                if cat != "none"
            ])
            st.bar_chart(df_prev.set_index("Category"))
    else:
        # Calculate from detections
        category_counts = {}
        for d in detections:
            cat = d.get("predicted_category", "none")
            if cat != "none":
                category_counts[cat] = category_counts.get(cat, 0) + 1

        total = len(detections)
        df_prev = pd.DataFrame([
            {"Category": cat, "Per 1000 Turns": 1000 * count / total}
            for cat, count in category_counts.items()
        ])
        if not df_prev.empty:
            st.bar_chart(df_prev.set_index("Category"))

    # Turn index analysis
    st.subheader("Detection Rate by Conversation Turn")

    turn_data = {}
    for d in detections:
        turn = d.get("turn_index", 0)
        if turn not in turn_data:
            turn_data[turn] = {"total": 0, "flagged": 0}
        turn_data[turn]["total"] += 1
        if d.get("predicted_category", "none") != "none":
            turn_data[turn]["flagged"] += 1

    df_turns = pd.DataFrame([
        {
            "Turn": turn,
            "Flag Rate": data["flagged"] / data["total"] if data["total"] > 0 else 0,
        }
        for turn, data in sorted(turn_data.items())
    ])

    if not df_turns.empty:
        st.line_chart(df_turns.set_index("Turn"))

    # Display figures if available
    st.subheader("Generated Figures")

    figures = list(FIGURES_DIR.glob("*.png"))
    if figures:
        cols = st.columns(2)
        for i, fig_path in enumerate(figures):
            with cols[i % 2]:
                st.image(str(fig_path), caption=fig_path.stem)
    else:
        st.info("No figures generated yet. Run analytics pipeline first.")


def render_reports_page():
    """Render reports page."""
    st.title("📋 Reports")

    reports = load_reports()

    # Gap report
    st.subheader("Benchmark vs Reality Gap Report")

    if "gap" in reports:
        gap = reports["gap"]
        summary = gap.get("summary", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("JS Divergence", f"{summary.get('js_divergence', 0):.4f}")
        with col2:
            st.metric("Spearman Correlation", f"{summary.get('spearman_correlation', 0):.4f}")
        with col3:
            st.metric("Samples Compared", f"{summary.get('wildchat_samples', 0):,}")

        # Interpretation
        interp = gap.get("interpretation", {})
        if interp:
            st.markdown("**Interpretation:**")
            for key, text in interp.items():
                st.markdown(f"- **{key}:** {text}")

        # Mismatches
        mismatches = gap.get("biggest_mismatches", [])
        if mismatches:
            st.markdown("**Biggest Mismatches:**")
            df_mismatch = pd.DataFrame(mismatches)
            st.dataframe(df_mismatch)
    else:
        st.info("Gap report not available. Run gap_report.py first.")

    st.markdown("---")

    # Reliability report
    st.subheader("Reliability Report")

    if "reliability" in reports:
        rel = reports["reliability"]
        summary = rel.get("summary", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Reliability Score", f"{summary.get('reliability_score', 0):.0%}")
        with col2:
            st.metric("Agreement Rate", f"{summary.get('agreement_rate', 0):.0%}")
        with col3:
            st.metric("Self-Consistency", f"{summary.get('self_consistency', 0):.0%}")

        # Recommendations
        recommendations = rel.get("recommendations", [])
        if recommendations:
            st.markdown("**Recommendations:**")
            for rec in recommendations:
                st.markdown(f"- {rec}")
    else:
        st.info("Reliability report not available. Run reliability_report.py first.")

    st.markdown("---")

    # Export section
    st.subheader("Export Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export Detections (CSV)"):
            detections = load_sample_data()
            df = pd.DataFrame(detections)
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "wildchat_detections.csv",
                "text/csv",
            )

    with col2:
        if st.button("Export Reports (JSON)"):
            reports_json = json.dumps(reports, indent=2)
            st.download_button(
                "Download JSON",
                reports_json,
                "wildguard_reports.json",
                "application/json",
            )


def main():
    """Main application entry point."""
    page = render_sidebar()

    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "🔍 Scan Transcript":
        render_scan_page()
    elif page == "📈 Analytics":
        render_analytics_page()
    elif page == "📋 Reports":
        render_reports_page()


if __name__ == "__main__":
    main()
