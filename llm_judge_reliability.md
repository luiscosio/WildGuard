# LLM-as-a-Judge Reliability (Notes + Action Items)

## Why this matters for WildGuard
WildGuard relies on LLM judgments to detect patterns in real chat logs. Recent work shows
single-shot LLM judgments can be inconsistent and misleading, even when settings are deterministic.
Therefore, we must treat "judge outputs" as probabilistic signals, not ground truth.

## Key takeaways (Schroeder & Wood-Doughty)
- Determinism improves consistency, not correctness.
- One sample from a stochastic model can be confidently wrong.
- Reliability should be estimated across multiple samples.
- Report agreement / uncertainty instead of only a final label.

## Practical implications for this repo
We should implement:
1) Multi-sample judging (self-consistency) as default mode for evaluation.
2) Agreement metrics (e.g., majority vote + disagreement rate).
3) Logging of per-sample votes so downstream analysis can audit reliability.

## Proposed MVP approach (low-risk)
Given a prompt, run the judge N times (e.g., N=5) and return:
- final_label: majority vote
- votes: list of labels
- disagreement_rate: 1 - (max_vote_count / N)

This makes uncertainty explicit and avoids silent single-shot failures.
