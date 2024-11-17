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

def drop_columns(df, cols):
    return df.drop(cols, axis=1)

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

            # Add a new column that merges release year with the title
            movies_subset_df['clean_title'] = movies_subset_df['title'] + movies_subset_df['release_date']
            movies_subset_df = self.feature_engineering(movies_subset_df)
            
            os.makedirs(os.path.dirname(self.ingestion.raw_data_path), exist_ok=True)
            movies_subset_df.to_csv(self.ingestion.raw_data_path, index=False, header=True)
            
            return movies_subset_df

        except Exception as e:
            return str(e)

    def feature_engineering(self, movies_subset_df):
        movies_subset_df = movies_subset_df.fillna('')
        cols_to_drop = ['title', 'release_date', 'cast']
        movies_subset_df = drop_columns(movies_subset_df, cols_to_drop)
        