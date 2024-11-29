#Import dependecies
import pandas as pd
import os

import joblib
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.components import data_transformation, movie_recommendation_by_title


def clean_release_date(date):
    return str(date).split('-')[0]

def drop_columns(df, cols):
    return df.drop(cols, axis=1)

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    combined_feature_path: str = os.path.join('artifacts', 'combined_data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            movies_df = pd.read_csv('src/data/movies.csv')

            # DataFrame cleaning
            # Create a subset  of the df since we won't be needing everything
            cleaned_df = self.feature_engineering(movies_df)
            # cleaned_df['release_date'] = cleaned_df['release_date'].apply(clean_release_date)

            # movies_subset_df = cleaned_df['genres'] + cleaned_df['keywords'] + \
            #                     cleaned_df['tagline'] + cleaned_df['cast'] + cleaned_df['director'] + cleaned_df['production_companies']
            
            os.makedirs(os.path.dirname(self.ingestion.raw_data_path), exist_ok=True)
            cleaned_df.to_csv(self.ingestion.raw_data_path, index=False, header=True)
            # movies_subset_df.to_csv(self.ingestion.combined_feature_path, index=False, header=True)

            return self.ingestion.raw_data_path

        except Exception as e:
            return str(e)

    def feature_engineering(self, movies_subset_df):
        selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
        for feat in selected_features:
            movies_subset_df[feat] = movies_subset_df[feat].fillna(' ')
        
        cols_to_drop = ['index', 'budget', 'homepage', 'id',
            'original_language', 'original_title', 'overview', 'popularity',
            'production_countries', 'release_date',
            'revenue', 'runtime', 'spoken_languages', 'status',
            'vote_average', 'vote_count', 'crew']
        movies_subset_df = drop_columns(movies_subset_df, cols_to_drop)
        return movies_subset_df
    
        

if __name__ == '__main__':
    obj = DataIngestion()
    cleaned_date_path = obj.initiate_data_ingestion()
    
    dt = data_transformation.DataTransformation()
    scaled_data = dt.initiate_data_transformation()
    movie_recom = movie_recommendation_by_title.MovieRecommendation()

    movie_title = 'three'

    recommendations = movie_recom.initiate_movie_recommendation(movie_title, scaled_data)
    print(recommendations)