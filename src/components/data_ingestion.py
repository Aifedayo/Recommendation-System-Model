#Import dependecies
import pandas as pd
import numpy as np
import difflib
import re
import os

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_release_date(date):
    return str(date).split('-')[0]

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            movies_df = pd.read_csv('src/data/movies.csv')

            # DataFrame cleaning
            # Create a subset  of the df since we won't be needing everything
            movies_subset_df = pd.concat([movies_df['genres'], movies_df['title'], movies_df['cast'], movies_df['release_date'], movies_df['director'],
                             movies_df['vote_average'], movies_df['popularity']], axis=1)

            movies_subset_df['release_date'] = movies_subset_df['release_date'].apply(clean_release_date)
            movies_subset_df = movies_subset_df.fillna('')
        except Exception as e:
            return str(e)
        


def recommend_movie(movie_name):
    pass