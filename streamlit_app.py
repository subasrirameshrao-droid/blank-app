import streamlit as st
import pickle
import re

# Load the trained AI model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Website title
st.title("🤖 AI-Based Spam Email Detection")

st.write("Detect spam, calculate risk, and understand why the email is suspicious.")

# Email input
email = st.text_area("📧 Enter your email:", height=200)

# Analyze button
if st.button("🔍 Analyze Email"):

    if email.strip():

        # Convert email into numbers
        email_vector = vectorizer.transform([email])

        # AI prediction
        prediction = model.predict(email_vector)[0]

        # Probability
        probabilities = model.predict_proba(email_vector)[0]
        spam_index = list(model.classes_).index("spam")
        spam_percentage = probabilities[spam_index] * 100

        # Show result
        if prediction == "spam":
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ NOT SPAM")

        # Show risk percentage
        st.subheader(f"Spam Risk: {spam_percentage:.1f}%")
        st.progress(int(spam_percentage))

        # Determine risk level
        if spam_percentage <= 30:
            risk_level = "Low"
        elif spam_percentage <= 60:
            risk_level = "Medium"
        elif spam_percentage <= 80:
            risk_level = "High"
        else:
            risk_level = "Very High"

        st.write(f"**Risk Level: {risk_level}**")

        # Find suspicious features
        reasons = []

        text = email.lower()

        urgent_words = [
            "urgent", "immediately", "act now",
            "limited time", "hurry", "verify now"
        ]

        prize_words = [
            "won", "winner", "prize", "reward",
            "lottery", "free money", "cash prize"
        ]

        information_words = [
            "password", "account", "bank",
            "credit card", "personal information",
            "verify your details"
        ]

        promotional_words = [
            "free", "offer", "discount",
            "bonus", "claim", "special offer"
        ]

        # Check for urgent language
        if any(word in text for word in urgent_words):
            reasons.append("⚠️ Urgent or pressure-based language detected")

        # Check for prize/reward language
        if any(word in text for word in prize_words):
            reasons.append("🎁 Prize or reward-related words detected")

        # Check for requests for information
        if any(word in text for word in information_words):
            reasons.append("🔐 Possible request for personal or financial information")

        # Check promotional language
        if any(word in text for word in promotional_words):
            reasons.append("📢 Promotional or suspicious offer detected")

        # Check links
        if re.search(r"http[s]?://|www\.", text):
            reasons.append("🔗 Link detected in the email")

        # Display explanation
        st.subheader("🔎 Why is this email suspicious?")

        if prediction == "spam":

            if reasons:
                for reason in reasons:
                    st.write(reason)
            else:
                st.write("⚠️ The AI model detected patterns associated with spam.")

        else:

            st.write("✅ No major suspicious patterns were detected.")

    else:
        st.warning("⚠️ Please enter an email first.")
