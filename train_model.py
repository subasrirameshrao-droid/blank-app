import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample training data
data = {
    "email": [
        "Congratulations! You won a free prize. Click the link now.",
        "You have won a lottery. Claim your reward immediately.",
        "Get a free gift by clicking this link.",
        "Urgent! Your account will be closed. Verify now.",
        "You have been selected for a special offer.",
        "Claim your free money today.",
        "Click here to receive your prize.",
        "Win a brand new phone. Apply now.",
        "Limited time offer! Get your reward now.",
        "You won a cash prize. Send your details to claim it.",

        "Please submit your assignment by tomorrow.",
        "The meeting is scheduled for 10 AM.",
        "Can you send me the project report?",
        "Your appointment is confirmed for Monday.",
        "Please find the notes from today's class.",
        "The project meeting will start at 2 PM.",
        "Don't forget to bring your documents tomorrow.",
        "Thank you for sending the report.",
        "Your order has been delivered successfully.",
        "Please contact me when you are available."
    ],

    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",

        "not spam", "not spam", "not spam", "not spam", "not spam",
        "not spam", "not spam", "not spam", "not spam", "not spam"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert email text into numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["email"])
y = df["label"]

# Train the AI model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("AI model trained successfully!")