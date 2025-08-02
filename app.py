import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Set Streamlit page config
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Get movie suggestions based on your favorite film!")


# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv(r"D:\Projects\Acemgrade\Movie Recommendation System\movies.csv")
    data.fillna("", inplace=True)
    data["combined_features"] = (
        data["genres"]
        + " "
        + data["keywords"]
        + " "
        + data["tagline"]
        + " "
        + data["cast"]
        + " "
        + data["director"]
    )
    return data


movies_data = load_data()

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
feature_vectors = vectorizer.fit_transform(movies_data["combined_features"])

# Cosine similarity matrix
similarity = cosine_similarity(feature_vectors)


# Recommendation function
def get_recommendations(movie_name):
    close_matches = difflib.get_close_matches(movie_name, movies_data["title"])
    if not close_matches:
        return None, None

    closest_match = close_matches[0]
    index_of_movie = movies_data[movies_data.title == closest_match].index[0]
    similarity_scores = list(enumerate(similarity[index_of_movie]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_titles = []
    for i in range(1, 11):  # Show top 10 recommendations
        index = sorted_movies[i][0]
        recommended_titles.append(movies_data.title[index])

    return closest_match, recommended_titles


# User input
movie_input = st.text_input("Enter a movie name")

if st.button("Recommend"):
    if movie_input.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        matched, recommendations = get_recommendations(movie_input)
        if not recommendations:
            st.error("❌ Movie not found. Please try another name.")
        else:
            st.success(f"Movies similar to **{matched}**:")
            for i, title in enumerate(recommendations, 1):
                st.write(f"{i}. {title}")
