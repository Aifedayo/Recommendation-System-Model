import os
import joblib
import difflib

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DataTransformationConfig:
    vectorizer_class: str = os.path.join('artifacts', 'vectorizer')


class DataTransformation:
    def __init__(self):
        self.transformation = DataTransformationConfig()

    def get_data_transformer(self):
        vectorizer = TfidfVectorizer(ngram_range=(1,6), stop_words='english', lowercase=True)
        return vectorizer