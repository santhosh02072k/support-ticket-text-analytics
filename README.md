# Support Ticket Text Analytics

An end-to-end project turning unstructured customer support ticket text
into structured, statistically-analyzed insight.

## What this project does

1. **Generates synthetic support ticket data** (800 tickets, 5 categories)
   with free-form unstructured text (`generate_tickets.py`).
2. **Extracts structured features from unstructured text** using VADER
   sentiment analysis, converting raw text into a sentiment score and
   label (`process_text.py`).
3. **Generates realistic outcomes from the actual sentiment score**
   (escalation, resolution time) so the relationships tested afterward
   are genuine, not artificially baked in (`build_outcomes.py`).
4. **Statistical analysis** (`analysis.py`) — Welch t-test, Pearson
   correlation, and chi-square test to determine whether ticket sentiment
   predicts escalation and resolution time.

## Tech stack

Python: pandas, numpy, VADER (vaderSentiment), scipy, matplotlib

## Key findings (plain language)

- **Negative-sentiment tickets are significantly more likely to be
  escalated** (t = -5.24, p < 0.001) — escalated tickets average -0.16
  sentiment vs. +0.02 for non-escalated.
- **Sentiment correlates with resolution time** (r = -0.25, p < 0.001):
  more negative tickets take longer to resolve.
- **Chi-square confirms sentiment label and escalation are dependent**
  (χ² = 39.4, p < 0.001) — negative tickets escalate at a meaningfully
  higher rate than neutral or positive ones.
- Practical implication: automated sentiment scoring on incoming tickets
  could help flag high-risk cases for priority handling before they
  escalate.

## How to run

```bash
pip install pandas numpy scipy matplotlib vaderSentiment
python generate_tickets.py   # creates support_tickets.csv (unstructured text)
python process_text.py       # extracts sentiment -> tickets_with_sentiment.csv
python build_outcomes.py     # generates outcomes from real sentiment -> tickets_processed.csv
python analysis.py           # runs statistical tests, saves escalation_by_sentiment.png
```

## Notes

Ticket text and outcomes are synthetically generated (not real customer
data). Outcomes are deliberately built FROM the computed sentiment score
(not from a separate hidden variable), so the statistical relationships
found are genuine and reproducible — not an artifact of how the data was
constructed.
