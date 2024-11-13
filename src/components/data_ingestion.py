#Import dependecies
import pandas as pd
import numpy as np
import difflib
import re
import os

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'movies.csv')
def recommend_movie(movie_name):
    