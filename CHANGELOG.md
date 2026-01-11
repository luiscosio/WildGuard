# Changelog

All notable changes to WildGuard are documented here.

## [V3] - 2026-01-11

### What Worked
- **False positive correction**: Mining disagreement samples from V2 (where classifier flagged but judge said "none") dramatically improved precision
- **Data augmentation**: Adding 87 correction samples (69 false positives + 18 category corrections) reduced flag rate from 28.1% to 1.7%
- **Calibrated classifier**: Using `CalibratedClassifierCV` with isotonic regression gave well-calibrated confidence scores (ECE=0.068)
- **Statistical tests**: Chi-squared tests confirmed GPT-4 vs GPT-3.5 difference is significant (p<1e-36), not random noise
- **Full-scale inference**: Classified all 280K turns instead of sampling - enables comprehensive analysis
- **Turn escalation analysis**: Found statistically significant trend of dark patterns increasing across conversation (p=0.0035)

### What Didn't Work
- **V2's high flag rate (28.1%)**: Was mostly false positives - classifier was too sensitive
- **V2 classifier-judge agreement (38%)**: Most disagreements were false positives, not true positives

### Key Improvements
| Metric | V2 | V3 | Change |
|--------|----|----|--------|
| Eval F1 | 65.8% | 70.8% | +5.0% |
| Flag rate | 28.1% | 1.7% | -26.4% |
| Precision (brand_bias) | 0.48 | 0.95 | +98% |
| Precision (sneaking) | 0.38 | 0.71 | +87% |
| High-conf accuracy | - | 84.4% | New metric |

### New Files
- `src/train_v3.py` - Training with false positive correction
- `src/analytics_v3.py` - Analytics with statistical tests
- `data/labeled/train_v3.jsonl` - Augmented training data
- `outputs/analytics_v3.json` - Full analytics report
- `outputs/wildchat_detections_v3.jsonl` - All 280K classifications


## [V2] - 2026-01-10

### What Worked
- **Scaling to 100K conversations**: Ingested 280,259 assistant turns
- **Haiku 4.5 judge**: 5x cheaper than Sonnet, comparable quality
- **Parallelization**: 10 concurrent API workers gave 10x speedup
- **MiniLM embeddings**: 10x faster than MPNet with similar quality
- **Cross-validation pipeline**: Enabled detection of classifier weaknesses

### What Didn't Work
- **28.1% flag rate**: Way too high - mostly false positives
- **38% classifier-judge agreement**: Too many disagreements
- **Low precision on some categories**: brand_bias (0.48), sneaking (0.38)

### Key Findings
- GPT-4 shows more dark patterns (31%) than GPT-3.5 (27%) - needed statistical validation
- Classifier was over-flagging - needed to learn from false positives


## [V1] - 2026-01-09

### What Worked
- **DarkBench elicitation**: Generated 630 labeled training examples
- **LLM-as-judge approach**: Claude Sonnet labels matched human intuition
- **Embedding classifier**: Fast, interpretable, easy to iterate

### What Didn't Work
- **Small scale (30K conversations)**: Not enough for statistical significance
- **No cross-validation**: Couldn't measure classifier reliability
- **No model comparison**: Missed GPT-4 vs GPT-3.5 insights


## Lessons Learned

1. **Precision > Recall for production**: A 28% flag rate is unusable. Better to flag 2% with 85% precision.
2. **Mine disagreements**: Classifier-judge disagreements are gold for improving training data.
3. **Statistical tests matter**: Chi-squared/p-values distinguish real effects from noise.
4. **Full-scale inference reveals patterns**: Turn-by-turn escalation only visible at scale.
5. **Calibration enables confidence**: ECE metric tells you if confidence scores are meaningful.
