import joblib
import streamlit as st

from src.components import (
    data_ingestion, data_transformation, movie_recommendation_by_title)

def movie_recommendation_by_title(input_data):
    obj = data_ingestion.DataIngestion()
    clean_data = obj.initiate_data_ingestion()

    dt = data_transformation.DataTransformation()
    movie_data, cleaned_data = dt.initiate_data_transformation(clean_data)

    movie_recom = movie_recommendation_by_title.MovieRecommendation()
    recommendations = movie_recom.initiate_movie_recommendation('iron man', movie_data)
    movie_recom = movie_recommendation_by_title.MovieRecommendation()
    recommendations = movie_recom.initiate_movie_recommendation('iron man', movie_data)
    movie_recommendation_by_title(input_data='Iron man')


def main():
    st.title('Movie Recommendations by Title')