import streamlit as st
import pickle
import re
import os
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# --- Build absolute paths to the model files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "src", "model")

tfidf = pickle.load(open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb"))
tfidf_matrix = pickle.load(open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), "rb"))
df = pickle.load(open(os.path.join(MODEL_DIR, "jobs_dataframe.pkl"), "rb"))

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def recommend_jobs(user_text, top_n=5):
    cleaned = clean_text(user_text)
    vector = tfidf.transform([cleaned])
    similarity = cosine_similarity(vector, tfidf_matrix)
    scores = similarity.flatten()
    top_indices = scores.argsort()[::-1][:top_n]
    return df.iloc[top_indices][["Job Title"]]

st.title("AI Job Recommendation System")

user_input = st.text_input("Enter your skills or resume text")

if st.button("Recommend Jobs"):
    results = recommend_jobs(user_input)
    st.write("Top Job Recommendations:")
    st.table(results)