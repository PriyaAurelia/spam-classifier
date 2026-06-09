"""
Spam Email Classifier
=====================
Author  : Priya Dharshini V
Tech    : Python · Scikit-learn · Naive Bayes · NLP · Matplotlib
Dataset : SMS Spam Collection (UCI) — loaded via sklearn / built-in sample

Pipeline:
  Raw Text → Preprocessing → TF-IDF Vectorisation
           → Multinomial Naive Bayes → Prediction + Evaluation
"""

import re
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# ── 1. Built-in sample dataset (no download needed) ───────────────────────────
# A representative sample of spam vs ham messages for demonstration.
MESSAGES = [
    # HAM (legitimate)
    ("ham",  "Hey, are you coming to the party tonight?"),
    ("ham",  "I will call you later, okay?"),
    ("ham",  "Can we meet tomorrow at the library?"),
    ("ham",  "Please remind me to submit the assignment by Friday."),
    ("ham",  "What time does the college bus leave?"),
    ("ham",  "Did you finish the maths problem set?"),
    ("ham",  "Let us catch up over coffee this weekend."),
    ("ham",  "Mom wants to know if you will be home for dinner."),
    ("ham",  "The project presentation is at 10 AM tomorrow."),
    ("ham",  "I just saw your message, will reply shortly."),
    ("ham",  "Happy birthday! Hope you have a wonderful day."),
    ("ham",  "Please send me the notes for today's class."),
    ("ham",  "I am running a bit late, start without me."),
    ("ham",  "The lab is open till 8 PM today."),
    ("ham",  "Can you share the meeting link?"),
    ("ham",  "Great work on the presentation today!"),
    ("ham",  "See you at the seminar at 3 PM."),
    ("ham",  "Just wanted to check if you received my email."),
    ("ham",  "We need to submit the form by end of day."),
    ("ham",  "Are you free for a quick call this afternoon?"),
    ("ham",  "The exam results will be out next week."),
    ("ham",  "Please review the document and give feedback."),
    ("ham",  "I will pick you up from the station at 6."),
    ("ham",  "Do not forget the team meeting at noon."),
    ("ham",  "Have a safe journey home!"),
    # SPAM
    ("spam", "Congratulations! You have won a FREE iPhone. Click now to claim!"),
    ("spam", "URGENT: Your bank account has been suspended. Verify immediately."),
    ("spam", "You are selected for a cash prize of Rs 50,000. Call now!"),
    ("spam", "FREE entry in our competition to win a holiday. Text WIN to 80085."),
    ("spam", "Earn Rs 5000 daily working from home. No experience needed. Join now!"),
    ("spam", "Your loan is approved! Get Rs 2 lakh instantly. Apply here."),
    ("spam", "LIMITED OFFER: 90% discount on all products today only! Shop now."),
    ("spam", "You have a pending KYC verification. Update now to avoid account block."),
    ("spam", "Win a brand new car! You have been selected. Confirm your details."),
    ("spam", "SIX chances to win CASH! Text YES to 98765 now."),
    ("spam", "Congratulations, your number was randomly selected for a reward!"),
    ("spam", "ALERT: Suspicious login detected. Click to secure your account now."),
    ("spam", "Make money fast! Join our exclusive investment group today."),
    ("spam", "Your OTP is expiring. Click link to validate your account immediately."),
    ("spam", "You owe taxes. Pay now to avoid legal action. IRS Notice 2024."),
    ("spam", "FREE gift card worth Rs 1000 waiting for you. Claim before midnight!"),
    ("spam", "Hot singles in your area are waiting to meet you. Sign up free!"),
    ("spam", "Exclusive deal just for you — unsubscribe not needed, act now!"),
    ("spam", "Your parcel could not be delivered. Pay Rs 50 to reschedule."),
    ("spam", "Double your investment in 7 days with our guaranteed crypto scheme!"),
]

# Extend dataset with augmented variations for better training
AUGMENTED = [
    ("ham",  "Let me know when you are free to discuss the project."),
    ("ham",  "The assignment deadline has been extended to Monday."),
    ("ham",  "Can you help me understand this concept from today's lecture?"),
    ("ham",  "I booked the tickets, we leave at 7 AM."),
    ("ham",  "Thank you for your help with the report!"),
    ("spam", "WINNER!! You have been chosen for our weekly prize draw. Claim now."),
    ("spam", "Loan approved in 5 minutes. No documents needed. Apply today."),
    ("spam", "Your Amazon account is locked. Click to verify identity immediately."),
    ("spam", "You have unused reward points expiring today. Redeem now!"),
    ("spam", "Act fast! Only 3 slots left in our wealth creation program."),
]

ALL_DATA = MESSAGES + AUGMENTED
labels   = [label for label, _ in ALL_DATA]
texts    = [text  for _, text  in ALL_DATA]


# ── 2. Text preprocessing ──────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

texts_clean = [preprocess(t) for t in texts]


# ── 3. Train / Test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    texts_clean, labels, test_size=0.25, random_state=42, stratify=labels
)


# ── 4. Pipeline: TF-IDF + Naive Bayes ─────────────────────────────────────────
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
    ("clf",   MultinomialNB(alpha=0.5))
])

model.fit(X_train, y_train)


# ── 5. Evaluation ─────────────────────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 55)
print("      SPAM EMAIL CLASSIFIER — Results")
print("=" * 55)
print(f"  Accuracy : {accuracy * 100:.1f}%")
print()
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))


# ── 6. Confusion Matrix plot ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Spam Email Classifier — Naive Bayes + TF-IDF",
             fontsize=14, fontweight="bold", color="#1A1A2E")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham ✉️", "Spam 🚫"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix", fontweight="bold")

# Class distribution bar chart
ham_count  = labels.count("ham")
spam_count = labels.count("spam")
bars = axes[1].bar(["Ham (Legitimate)", "Spam"], [ham_count, spam_count],
                   color=["#3B82F6", "#EF4444"], width=0.5, edgecolor="white")
axes[1].set_title("Dataset Distribution", fontweight="bold")
axes[1].set_ylabel("Number of Messages")
axes[1].set_ylim(0, max(ham_count, spam_count) + 5)
for bar, val in zip(bars, [ham_count, spam_count]):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5, str(val),
                 ha="center", fontweight="bold", fontsize=12)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
print("  Chart saved → results.png")
plt.show()


# ── 7. Interactive prediction demo ────────────────────────────────────────────
TEST_MESSAGES = [
    "Congratulations! You won a free iPhone. Click now!",
    "Hey, are you coming to the study group tonight?",
    "URGENT: Your bank account is locked. Verify now.",
    "Let us meet at the library tomorrow at 10 AM.",
    "You have been selected for a Rs 50,000 cash prize!",
    "Please send me the notes from today's class.",
]

print("\n" + "=" * 55)
print("      LIVE PREDICTION DEMO")
print("=" * 55)
for msg in TEST_MESSAGES:
    prediction = model.predict([preprocess(msg)])[0]
    prob       = model.predict_proba([preprocess(msg)])[0]
    confidence = max(prob) * 100
    tag = "🚫 SPAM" if prediction == "spam" else "✉️  HAM "
    print(f"  {tag}  ({confidence:5.1f}%)  →  {msg[:55]}...")
print("=" * 55)
