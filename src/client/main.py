import joblib
import streamlit as st
import pandas as pd
import numpy as np

from src.components import (
    data_ingestion, data_transformation, movie_recommendation_by_title)

def movie_recommend_by_title(movie_title):
    obj = data_ingestion.DataIngestion()
    clean_data = obj.initiate_data_ingestion()

    dt = data_transformation.DataTransformation()
    scaled_data = dt.initiate_data_transformation(clean_data)

    movie_recom = movie_recommendation_by_title.MovieRecommendation()
    recommendations = movie_recom.initiate_movie_recommendation(movie_title, scaled_data)

    return recommendations

def main():
    st.title('Movie Recommendations by Title')

    movie_name = st.text_input('Enter a movie title')
    recom = ''
    if st.button('Recommend'):
        recom = movie_recommend_by_title(movie_name)
        st.success('Here are your recommendations')
        movies_dict = {'movies': recom}
        df = pd.DataFrame(data=movies_dict, index=np.arange(1, 11))
        st.dataframe(df)


if __name__ == '__main__':
    main()
