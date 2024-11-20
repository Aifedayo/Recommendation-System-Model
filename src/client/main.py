import joblib
import streamlit as st

from src.components import (
    data_ingestion, data_transformation, movie_recommendation_by_title)

def movie_recommend_by_title(input_data):
    obj = data_ingestion.DataIngestion()
    clean_data = obj.initiate_data_ingestion()

    dt = data_transformation.DataTransformation()
    movie_data, cleaned_data = dt.initiate_data_transformation(clean_data)

    movie_recom = movie_recommendation_by_title.MovieRecommendation()
    recommendations = movie_recom.initiate_movie_recommendation(input_data, movie_data)

    return recommendations

def main():
    st.title('Movie Recommendations by Title')

    movie_name = st.text_input('Enter a movie title')
    recom = ''
    if st.button('Movie Recommendation'):
        recom = movie_recommend_by_title(movie_name)
        recom = recom.values
        st.success('Here are your recommendations')
    st.write(recom)


if __name__ == '__main__':
    main()
