import os
import pandas as pd
import joblib

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommendation:
    def initiate_movie_recommendation(self, title):
        vectorizer = joblib.load('artifacts/vectorizer')
        title = vectorizer.transform(title)