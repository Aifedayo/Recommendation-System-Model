import os
import pandas as pd
import numpy as np
import joblib

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommendation:
    def initiate_movie_recommendation(self, title, movies_data):
        vectorizer = joblib.load('artifacts/vectorizer')
        title = vectorizer.transform([title])

        scaled_data_feature = joblib.load('artifacts/scaled_data')
        similarity = cosine_similarity(title, scaled_data_feature).flatten()

        indices = np.argsort(similarity)[-10:]
        results = movies_data.iloc[indices]['clean_title'][::-1]
        return results