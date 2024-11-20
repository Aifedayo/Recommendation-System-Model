import os
import joblib
import pickle

import pandas as pd

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from src.utils import save_object

def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)

    except Exception as e:
        return str(e)

@dataclass
class DataTransformationConfig:
    vectorizer_path: str = os.path.join('artifacts', 'vectorizer')
    scaled_data_path: str = os.path.join('artifacts', 'scaled_data')


class DataTransformation:
    def __init__(self):
        self.transformation = DataTransformationConfig()

    def get_data_transformer(self):
        vectorizer = TfidfVectorizer(ngram_range=(1,6), stop_words='english', lowercase=True)
        save_object(self.transformation.vectorizer_path, vectorizer)
        return vectorizer
    

    def initiate_data_transformation(self, data_path):
        data = pd.read_csv(data_path)

        vectorizer = self.get_data_transformer()

        scaled_data = vectorizer.fit_transform(data['clean_title'])
        joblib.dump(vectorizer, self.transformation.vectorizer_path)
        joblib.dump(scaled_data, self.transformation.scaled_data_path)
        return scaled_data
    