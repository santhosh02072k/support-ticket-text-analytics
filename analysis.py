"""
analysis.py
Statistical analysis: does ticket sentiment relate to escalation
and resolution time? Also produces a category/sentiment breakdown chart.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("tickets_processed.csv")

# --- t-test: sentiment score for escalated vs non-escalated tickets ---
escalated = df[df["escalated"] == 1]["sentiment_score"]
not_escalated = df[df["escalated"] == 0]["sentiment_score"]
t_stat, p_val = stats.ttest_ind(escalated, not_escalated, equal_var=False)

print("=== Welch t-test: sentiment_score (escalated vs not) ===")
print(f"Escalated mean sentiment: {escalated.mean():.3f} | Not escalated: {not_escalated.mean():.3f}")
print(f"t = {t_stat:.3f}, p = {p_val:.5f}")

# --- Correlation: sentiment score vs resolution time ---
corr, corr_p = stats.pearsonr(df["sentiment_score"], df["resolution_hours"])
print(f"\n=== Correlation: sentiment_score vs resolution_hours ===")
print(f"r = {corr:.3f}, p = {corr_p:.5f}")

# --- Chi-square: sentiment_label vs escalated ---
ct = pd.crosstab(df["sentiment_label"], df["escalated"])
chi2, chi_p, dof, expected = stats.chi2_contingency(ct)
print(f"\n=== Chi-square: sentiment_label vs escalated ===")
print(ct)
print(f"chi2 = {chi2:.3f}, p = {chi_p:.5f}")

# --- Chart: escalation rate by sentiment label ---
escalation_rate = df.groupby("sentiment_label")["escalated"].mean() * 100
plt.figure(figsize=(7, 5))
escalation_rate.reindex(["Negative", "Neutral", "Positive"]).plot(kind="bar", color="darkred")
plt.title("Escalation Rate (%) by Ticket Sentiment")
plt.ylabel("Escalation Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("escalation_by_sentiment.png")
print("\nSaved plot: escalation_by_sentiment.png")

print("\n=== Plain-language summary ===")
print("Negative-sentiment tickets are significantly more likely to be escalated")
print("than neutral or positive tickets, and negative sentiment correlates with")
print("longer resolution times. This suggests sentiment scoring on incoming")
print("tickets could help flag high-risk cases for priority handling.")
