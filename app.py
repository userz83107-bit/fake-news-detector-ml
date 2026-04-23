import streamlit as st
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    fake = pd.read_csv("https://raw.githubusercontent.com/YOUR_USERNAME/fake-news-detector/main/Fake.csv")
true = pd.read_csv("https://raw.githubusercontent.com/YOUR_USERNAME/fake-news-detector/main/True.csv")
    return fake, true

fake, true = load_data()

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

# Split
X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# UI
st.title("📰 Fake News Detection System")
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

st.subheader("Dataset Preview")
st.write(data.head())

input_text = st.text_area("Enter News Text")

if st.button("Check News"):
    vector_input = vectorizer.transform([input_text])
    prediction = model.predict(vector_input)

    if prediction[0] == 1:
        st.success("✅ This looks like REAL news")
    else:
        st.error("❌ This looks like FAKE news")