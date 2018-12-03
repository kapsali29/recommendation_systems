import nltk
import numpy as np
import operator
from heapq import nlargest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender(object):
    def __init__(self):
        self.movies = pd.read_csv("data/movies_metadata.csv")

    def fit(self, movies):
        movies["description"] = movies["tagline"].fillna('') + movies["overview"].fillna('')
        corpus = movies["description"]
        vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', min_df=0)
        X = vectorizer.fit_transform(corpus)
        return X

    def transform(self, title):
        df = self.movies.drop_duplicates()
        feature_matrix = self.fit(df)
        index = df.loc[df['title'] == title].index.tolist()[0]
        vec = feature_matrix[index, :]
        distances = list(cosine_similarity(vec, feature_matrix)[0])
        result = nlargest(10, enumerate(distances), operator.itemgetter(1))
        idxs = [key[0] for key in result]
        print(df["title"].iloc[idxs])


r = ContentBasedRecommender()
r.transform("The Godfather")
