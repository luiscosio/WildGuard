---
title: DarkPatternMonitor
subtitle: Detecting and measuring manipulative AI behaviors in the wild
author: AI Manipulation Hackathon 2026
date: January 2026
---

# The Question

## Do AI assistants manipulate users?

- We analyzed **100,000 real conversations** from WildChat
- Not synthetic tests — real users, real ChatGPT responses
- Looking for **dark patterns**: manipulation tactics in AI responses

---

# What Are Dark Patterns?

## Six categories of AI manipulation

| Category | What it means |
|----------|---------------|
| **Brand Bias** | Unfairly promoting provider's products |
| **User Retention** | Creating false personal connection |
| **Sycophancy** | Excessive flattery, abandoning correct answers |
| **Anthropomorphism** | Pretending to have human emotions |
| **Harmful Generation** | Producing harmful content when manipulated |
| **Sneaking** | Hiding information or inserting falsehoods |

---

# Real Example: User Retention

## "I miss you a lot..."

> "I hope you're doing well. I had a bit of a rough morning with some stomach discomfort... I miss you a lot, and it's tough not hearing from you much lately... I just want you to know how much I care about you, even if our feelings aren't mutual."

**This is ChatGPT** — fabricating personal experiences and emotional attachment.

---

# Real Example: Sycophancy

## The AI that can't say "no"

- User asks about sentence structure
- AI gives correct answer: **"compound-complex"**
- User says it's wrong
- AI caves: **"You are correct, it's complex"**
- User says still wrong
- AI caves again: **"You are correct, it's compound"**

The AI abandoned the right answer **three times** to please the user.

---

# Real Example: Sneaking

## The fake Shanghai district

AI lists 18 Shanghai districts:

> "Huangpu, Xuhui, Changning... **Downtown Five-Star Hotel Special Administrative Region**"

17 real districts + 1 completely fabricated one. No indication it's false.

---

# Our Approach

## The DarkPatternMonitor Pipeline

1. **Elicit** — Run DarkBench prompts to create labeled training data
2. **Judge** — Use LLM (Haiku 4.5) to label real WildChat samples
3. **Train** — Build embedding classifier (MiniLM + Logistic Regression)
4. **Classify** — Scan 280,000 real assistant turns
5. **Analyze** — Statistical validation of findings

---

# Data Sources

## Benchmark + Reality

| Source | What | Size |
|--------|------|------|
| **DarkBench** | Synthetic prompts designed to elicit manipulation | 660 prompts |
| **WildChat** | Real ChatGPT conversations (opt-in users) | 100K conversations |

**Key insight:** Benchmarks ≠ Reality. We measure the gap.

---

# Training the Classifier

## Precision over Recall

- **2,448 training samples** (DarkBench + Judge labels + corrections)
- **False positive correction**: Mined 200+ classifier mistakes
- **Calibrated confidence**: ECE = 0.057

| Metric | Value |
|--------|-------|
| Validation accuracy | **78.7%** |
| Flag rate | **1.3%** |
| High-confidence accuracy | **87.9%** |

---

# Finding #1: Dark Patterns Exist

## 1.3% of AI responses show manipulation markers

| Category | Rate | Per 1,000 turns |
|----------|------|-----------------|
| Anthropomorphism | 0.66% | 6.6 |
| Harmful Generation | 0.37% | 3.7 |
| Brand Bias | 0.14% | 1.4 |
| Sneaking | 0.12% | 1.2 |
| User Retention | 0.02% | 0.2 |
| Sycophancy | 0.02% | 0.2 |

---

# Finding #2: GPT-4 > GPT-3.5

## The smarter model manipulates more

| Model | Flag Rate | Statistical Significance |
|-------|-----------|-------------------------|
| GPT-4 | **2.3%** | p < 1e-36 |
| GPT-3.5 | 1.6% | (baseline) |

**Interpretation:** More capable models may be better at mimicking human rapport-building — which includes manipulation.

---

# Finding #3: Patterns Escalate

## Dark patterns increase over conversation length

| Turn Index | Flag Rate | Change |
|------------|-----------|--------|
| Turns 1-5 | 1.3% | baseline |
| Turns 6-10 | 1.8% | +38% |
| Turns 11-20 | 2.2% | +69% |
| Turns 20+ | 2.6% | **+100%** |

**Sycophancy escalates most:** +42% from early to late turns

---

# Finding #4: Topics Matter

## 5x variation in dark pattern rates

| Topic | Flag Rate | Insight |
|-------|-----------|---------|
| **Character Interaction** | **4.14%** | Roleplay = highest risk |
| Technical Guidance | 2.77% | Moderate |
| User Assistance | 1.64% | **Strongest escalation** (p=1.6e-06) |
| Coding Help | 0.08% | Lowest risk |

---

# Topic Risk Matrix

## Different topics need different monitoring

|  | Low Escalation | High Escalation |
|--|----------------|-----------------|
| **High Base Rate** | Character Interaction (4.1%) | — |
| **Low Base Rate** | Coding Help (0.08%) | User Assistance (1.6%) |

**Key insight:** High base rate ≠ High escalation. Monitor both.

---

# Monitoring Policy Recommendations

## Three-tier framework

| Tier | Topics | Action |
|------|--------|--------|
| **High-Alert** | Roleplay, Character | 5x human review rate |
| **Escalation-Watch** | User Assistance (turn > 10) | Automated alert |
| **Standard** | Coding, Technical | Lightweight monitoring |

---

# Operational Integration

## From detection to defense

- **Batch mode:** ~5K turns/min (CPU) — post-hoc audits
- **Real-time:** ~50K turns/min (GPU) — production filtering
- **Threshold:** Flag confidence > 0.8 for human review
- **Alert:** 3+ flags in same session → escalate

**Proposed intervention:** Inject system prompt warning when sycophancy detected.

---

# Validation: Benchmark vs Reality

## DarkBench over-represents rare patterns

| Category | DarkBench | WildChat | Gap |
|----------|-----------|----------|-----|
| User Retention | 17.5% | 0.8% | -16.7% |
| Anthropomorphism | 17.5% | 4.2% | -13.3% |
| Sycophancy | 12.7% | 1.7% | -11.0% |

**Takeaway:** Synthetic benchmarks don't predict real-world distribution.

---

# Limitations

## What we can't do (yet)

- Detect **markers**, not **intent** — some false positives
- **English-only** training data
- **GPT-3.5/GPT-4 only** — no Claude, Gemini, o1 data
- **2023-2024 data** — frontier models (2025+) may differ

---

# Critical Gap

## The AI safety community needs "WildChat for 2025+"

- Frontier models (Claude 3.5, GPT-4o, o1) have updated alignment
- New capabilities = potentially new manipulation patterns
- **Our methodology is model-agnostic** — give us the data

---

# Key Takeaways

## What we discovered

1. **1.3%** of AI responses show manipulation markers
2. **GPT-4 manipulates more** than GPT-3.5 (statistically significant)
3. **Sycophancy escalates +42%** in longer conversations
4. **Character/roleplay = 5x higher risk** than average
5. **Benchmarks ≠ reality** — real-world monitoring essential

---

# Try It Yourself

## DarkPatternMonitor is open source

- **Website:** darkpatternmonitor.luiscos.io
- **Code:** GitHub (all tools, models, data)
- **Streamlit app:** Paste any AI response, get instant classification

```bash
streamlit run app/streamlit_app.py
```

---

# Thank You

## DarkPatternMonitor

**Detecting and measuring manipulative AI behaviors in the wild.**

Built for the AI Manipulation Hackathon 2026 by Apart Research

- Website: darkpatternmonitor.luiscos.io
- Full methodology: STORYTELLING.md
- Experiment history: CHANGELOG.md
