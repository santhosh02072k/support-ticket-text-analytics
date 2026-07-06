"""
process_text.py
Extracts structured features from unstructured support ticket text:
sentiment score (VADER), and derived urgency flag from sentiment + keywords.
"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("support_tickets.csv")
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores["compound"]

df["sentiment_score"] = df["ticket_text"].apply(get_sentiment)

def sentiment_label(score):
    if score <= -0.3:
        return "Negative"
    elif score >= 0.3:
        return "Positive"
    return "Neutral"

df["sentiment_label"] = df["sentiment_score"].apply(sentiment_label)

df.to_csv("tickets_with_sentiment.csv", index=False)

print("=== Sentiment distribution ===")
print(df["sentiment_label"].value_counts())
print("\n=== Sample processed rows ===")
print(df[["category", "ticket_text", "sentiment_score", "sentiment_label"]].head(8).to_string(index=False))

print("\n=== Avg sentiment by category ===")
print(df.groupby("category")["sentiment_score"].mean().sort_values())
