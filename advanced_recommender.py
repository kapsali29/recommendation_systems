"""
We are going to increase the quality of the recommender by building better metadata. More specifically we will use
the top 3 actors, the director, related genres and the movie plot keywords
"""
from heapq import nlargest

import pandas as pd
import ast
import operator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class AdvancedRecommender(object):
    def __init__(self):
        self.keywords = pd.read_csv("data/keywords.csv")
        self.credits = pd.read_csv("data/credits.csv")
        self.metadata = pd.read_csv("data/movies_metadata.csv")
        self.movies_info = pd.DataFrame()

    def find_movie_keywords(self):
        self.keywords["keys"] = self.keywords["keywords"].apply(
            lambda x: [a["name"] for a in ast.literal_eval(x)] if ast.literal_eval(x) != [] else [])
        pass

    def find_cast_and_crew(self):
        self.credits["top_actors"] = self.credits["cast"].apply(
            lambda x: [actor["name"] for actor in ast.literal_eval(x)[:3]] if ast.literal_eval(x) != [] else [])
        self.credits["director"] = self.credits["crew"].apply(
            lambda x: [credit["name"] for
                       credit in ast.literal_eval(x) if
                       credit["department"] == "Directing" and credit["job"] == "Director"] if ast.literal_eval(
                x) != [] else [])
        pass

    def find_movie_genres(self):
        self.metadata["movie_types"] = self.metadata["genres"].apply(
            lambda x: [genre["name"] for genre in ast.literal_eval(x)] if ast.literal_eval(x) != [] else [])
        pass

    def create_new_df(self):
        self.movies_info["title"] = self.metadata["title"]
        self.movies_info["movie_types"] = self.metadata["movie_types"]
        self.movies_info["director"] = self.credits["director"]
        self.movies_info["top_actors"] = self.credits["top_actors"]
        self.movies_info["keys"] = self.keywords["keys"]
        pass

    def summarize_info(self, movie):
        title = movie["title"].replace(" ", "")
        types = [type.replace(" ", "") for type in movie["movie_types"]]
        director = [dir.replace(" ", "") for dir in movie["director"]]
        actors = [actor.replace(" ", "") for actor in movie["top_actors"]]
        keys = [key.replace(" ", "") for key in movie["keys"]]
        return title.lower() + " " + " ".join(types).lower() + " " + " ".join(
            director).lower() + " " + " ".join(
            actors).lower() + " " + " ".join(keys).lower()

    def fit(self, movie):
        vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', min_df=0)
        corpus = movie["soup"]
        X = vectorizer.fit_transform(corpus)
        return X

    def transform(self, title):
        df = self.movies_info
        feature_matrix = self.fit(df)
        index = df.loc[df['title'] == title].index.tolist()[0]
        vec = feature_matrix[index, :]
        distances = list(cosine_similarity(vec, feature_matrix)[0])
        result = nlargest(10, enumerate(distances), operator.itemgetter(1))
        idxs = [key[0] for key in result]
        print(df["title"].iloc[idxs])

    def main(self, title):
        print(" Extract movies keywords")
        self.find_movie_keywords()
        print(" Extract top 3 actors and directors")
        self.find_cast_and_crew()
        print(" Extract movie genres")
        self.find_movie_genres()
        print(" Summarize the info to one data frame")
        self.create_new_df()
        self.movies_info = self.movies_info.fillna(" ")
        self.movies_info["soup"] = self.movies_info.apply(self.summarize_info, axis=1)
        print(" Get feature matrix")
        print(self.movies_info)
        print(" Find {} neighbors".format(title))
        self.transform(title)


a = AdvancedRecommender()
a.main('The Godfather')
