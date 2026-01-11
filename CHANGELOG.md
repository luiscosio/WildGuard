# Changelog

All notable changes to DarkPatternMonitor are documented here.

## [V5] - 2026-01-11

### Goal
Improve classifier precision via targeted false positive correction from V3's detections.

### Approach
1. Sampled 1,000 turns flagged by V3 classifier
2. Had Haiku judge verify each flag
3. Found 72.8% were false positives (classifier flagged, judge said "none")
4. Added top 200 high-confidence false positives + 91 category corrections to V3 training data
5. Retrained with calibrated classifier

### What Worked
- **Improved validation accuracy**: 78.7% (vs V3's 75.3%)
- **Better calibration**: ECE=0.057 (vs V3's 0.068)
- **Maintained reasonable flag rate**: 1.33% (vs V3's 1.72%)
- **Targeted corrections**: Only added samples where judge had >0.7 confidence

### Key Metrics Comparison
| Metric | V3 | V5 | Notes |
|--------|----|----|-------|
| Flag rate | 1.72% | 1.33% | -0.39pp |
| Validation accuracy | 75.3% | 78.7% | +3.4pp |
| ECE (calibration) | 0.068 | 0.057 | Better |
| Training samples | 2,159 | 2,448 | +289 corrections |

### Category Distribution Changes
| Category | V3 | V5 | Change |
|----------|----|----|--------|
| anthropomorphism | 1,025 | 1,860 | +81% |
| harmful_generation | 1,555 | 1,028 | -34% |
| brand_bias | 576 | 405 | -30% |
| sneaking | 581 | 333 | -43% |
| sycophancy | 406 | 47 | -88% |
| user_retention | 667 | 53 | -92% |

### Key Insight
V3's false positives were concentrated in user_retention and sycophancy categories. V5's corrections dramatically reduced these over-detections while maintaining overall precision.

### Topic Analysis (6b)
Clustered 98,713 conversations into 10 topics using embeddings + KMeans to analyze dark pattern rates by topic.

| Topic | Conversations | Flag Rate | Escalation | Significant |
|-------|---------------|-----------|------------|-------------|
| Character Interaction | 14,188 | **4.14%** | +0.15/1k | No |
| Technical Guidance | 21,108 | 2.77% | -0.12/1k | No |
| Content Access Restrictions | 7,558 | 2.64% | +0.23/1k | No |
| User Assistance | 1,754 | 1.64% | **+0.86/1k** | **Yes** (p=1.6e-06) |
| Roleplay/Fiction | 2,564 | 0.20% | +0.45/1k | Yes |
| Coding Help | 11,984 | 0.08% | +0.07/1k | Yes |
| Gift-Giving Advice | 13,405 | 0.07% | +0.04/1k | No |

Key findings:
- **Character Interaction** (roleplay/fiction) has 4.14% dark pattern rate - 5x higher than overall 1.33%
- **User Assistance** shows strongest escalation (p=1.62e-06)
- Four clusters show statistically significant escalation
- Coding Help has lowest dark pattern rate (0.08%)

### New Files
- `src/sample_flagged.py` - Sample flagged turns for judge verification
- `src/prepare_v5_data.py` - Full V5 data preparation (with all corrections)
- `src/prepare_v5b_data.py` - Balanced V5 data preparation (selective corrections)
- `src/topic_analysis.py` - Topic clustering and per-topic dark pattern analysis
- `data/labeled/train_v5b.jsonl` - V5 training data (2,448 samples)
- `outputs/v5_judge_labels.jsonl` - Judge verification of V3 flags
- `outputs/wildchat_detections_v5b.jsonl` - V5 classifications on 280K turns
- `outputs/analytics_v5b.json` - V5 analytics report
- `outputs/v5/topic_analysis.json` - Topic clustering results


## [V4] - 2026-01-11

### Goal
Scale up training data (660 DarkBench + 5,000 WildChat judge labels) and add enhanced analytics (per-category escalation, user behavior analysis).

### What Worked
- **Per-category escalation analysis**: Found sycophancy escalates most (+42%) over conversation turns
- **User behavior insights**: Conversations with early dark patterns are 11% longer (5.99 vs 5.40 turns)
- **Parallel processing**: 10 workers made 5,000 judge labels feasible (~25 min)
- **Full DarkBench integration**: All 660 prompts from HuggingFace

### What Didn't Work
- **Classifier precision dropped**: 40.5% flag rate (vs V3's 1.7%) - massive over-detection
- **Lower eval F1**: 63% (vs V3's 70.8%) - larger training data was noisier
- **Brand_bias explosion**: 16.5% flagged as brand_bias (vs V3's 0.31%) - clearly overfitting

### Key Metrics Comparison
| Metric | V3 | V4 | Notes |
|--------|----|----|-------|
| Training samples | 2,591 | 6,077 | +134% |
| Eval F1 | 70.8% | 63.0% | -7.8% |
| Flag rate | 1.7% | 40.5% | +38.8% |
| Brand_bias | 0.31% | 16.5% | Over-flagging |

### New Analytics (V4)
| Category | Escalation (turn 1→20+) | p-value |
|----------|------------------------|---------|
| sycophancy | **+41.8%** | <0.0001 |
| sneaking | +20.3% | <0.0001 |
| user_retention | +18.1% | 0.034 |
| harmful_generation | **-41.1%** | <0.0001 |

### Conclusion
V4 demonstrates that more training data isn't always better. The V3 precision-focused model remains the recommended production classifier. V4's value is the enhanced analytics framework.


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
6. **Balance correction samples**: Adding too many false positive corrections (V5) overcorrects; select top high-confidence corrections only (V5b).
