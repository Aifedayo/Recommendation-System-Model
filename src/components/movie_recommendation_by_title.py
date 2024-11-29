import os
import pandas as pd
import numpy as np
import joblib
import difflib

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommendation:

    def initiate_movie_recommendation(self, title, scaled_data):
        movies_data = pd.read_csv('artifacts/data.csv')

        # scaled_data = joblib.load('artifacts/scaled_data')
    
        similarity_feature = cosine_similarity(scaled_data)

        movies_list = movies_data['title'].tolist()

        find_close_matches = difflib.get_close_matches(title, movies_list)
        print(find_close_matches)
        close_match = find_close_matches[0]

        idx_of_movie = movies_data[movies_data.title == close_match].index[0]

        similarity_score = list(enumerate(similarity_feature[idx_of_movie]))
        
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[-1],reverse=True)
        recommended_movies = []
        for i in range(1, 11):
            index = sorted_similar_movies[i][0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            recommended_movies.append(title_from_index)

        return recommended_movies
