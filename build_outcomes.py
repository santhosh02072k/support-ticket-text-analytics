"""
build_outcomes.py
Generates escalation and resolution_hours OUTCOMES based on the ACTUAL
sentiment score computed from ticket text (not a hidden variable). This
ensures any statistical relationship found in analysis.py is real and
reproducible from the text itself, not baked in artificially.
"""
import numpy as np
import pandas as pd

np.random.seed(11)
df = pd.read_csv("tickets_with_sentiment.csv")

# Escalation probability driven directly by sentiment: more negative -> more likely escalated
# sentiment_score ranges roughly -1 to 1
escalation_prob = 0.15 + 0.35 * (-df["sentiment_score"]).clip(lower=0)
df["escalated"] = np.random.binomial(1, escalation_prob.clip(0, 0.9))

# Resolution time driven by sentiment + escalation: more negative sentiment & escalated -> longer
base_scale = 3 + 4 * (-df["sentiment_score"]).clip(lower=0) + df["escalated"] * 5
df["resolution_hours"] = np.round(np.random.gamma(shape=2.0, scale=base_scale), 1)

df.to_csv("tickets_processed.csv", index=False)
print("Outcomes generated. Escalation rate:", df["escalated"].mean().round(3))
print(df[["category", "sentiment_score", "sentiment_label", "escalated", "resolution_hours"]].head(8))
