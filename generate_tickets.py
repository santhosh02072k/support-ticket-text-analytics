"""
generate_tickets.py
Generates synthetic (but realistic) customer support ticket data with
free-form unstructured text, mirroring real support ticket patterns.
Escalation and resolution time are generated in build_outcomes.py, AFTER
sentiment is computed from the actual text - so any relationship found
later is real and reproducible, not baked in from a hidden variable.
"""
import numpy as np
import pandas as pd

np.random.seed(7)
N = 800

categories = ["Billing", "Technical Issue", "Account Access", "Shipping/Delivery", "General Inquiry"]

templates = {
    "Billing": [
        "I was charged twice for my subscription this month, please refund the extra charge.",
        "My invoice shows an incorrect amount, can someone check this?",
        "Why is my bill higher than usual? I did not change my plan.",
        "I need a receipt for last month's payment for my records.",
        "This is the third time I've been overcharged. Extremely frustrating.",
    ],
    "Technical Issue": [
        "The app keeps crashing every time I try to log in, this is really annoying.",
        "I can't upload files anymore, getting an error message constantly.",
        "The website has been down for me since this morning.",
        "Login page won't load, tried on two browsers already.",
        "Nothing is working today, this is the worst experience I've had.",
    ],
    "Account Access": [
        "I forgot my password and the reset link isn't arriving in my email.",
        "My account got locked after too many login attempts, please help.",
        "I can't access my account from my new phone.",
        "Someone please unlock my account, I've been trying for an hour.",
        "Two-factor authentication code never arrives, I'm stuck.",
    ],
    "Shipping/Delivery": [
        "My order was supposed to arrive three days ago and there's no update.",
        "The package arrived damaged, very disappointed with the packaging.",
        "Tracking says delivered but I never received anything.",
        "Can you tell me when my order will actually ship?",
        "This delay is unacceptable, I needed this for an event.",
    ],
    "General Inquiry": [
        "Just wanted to ask about your return policy before I buy.",
        "Do you offer student discounts on annual plans?",
        "Thanks for the quick help earlier, just confirming the update went through.",
        "What are your customer support hours on weekends?",
        "Loved the recent update, just wanted to say the new dashboard looks great.",
    ],
}

rows = []
for i in range(N):
    cat = np.random.choice(categories)
    text = np.random.choice(templates[cat])
    rows.append({"ticket_id": f"TICKET{i:05d}", "category": cat, "ticket_text": text})

df = pd.DataFrame(rows)
df.to_csv("support_tickets.csv", index=False)
print(f"Generated {len(df)} tickets across {df['category'].nunique()} categories")
print(df.head())
