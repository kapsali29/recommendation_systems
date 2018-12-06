import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


class UserBasedCFMovies(object):
    def __init__(self):
        self.ratings = pd.read_csv("data/ratings_small.csv")
        self.users = self.ratings["userId"].nunique()
        self.movies = self.ratings["movieId"].nunique()
        self.users_movies = np.zeros((self.users, self.movies))
        self.id_mapper = {}

    def transform_ids(self):
        movieIds = list(self.ratings["movieId"].unique())
        correct_ids = [x for x in movieIds if self.movies >= x]
        correct_ids_len = len(correct_ids)
        for movieId in movieIds:
            if movieId not in correct_ids:
                self.id_mapper[movieId] = correct_ids_len
                correct_ids_len += 1
        self.ratings["movieId"] = self.ratings["movieId"].apply(
            lambda x: self.id_mapper[x] if x in self.id_mapper.keys() else x)
        pass

    def construct_array(self):
        for line in self.ratings.itertuples():
            self.users_movies[line[1] - 1, line[2] - 1] = line[3]
        cosine_matrix = 1 - pairwise_distances(self.users_movies, metric="cosine")
        return cosine_matrix

    def predict(self):
        sims = self.construct_array()
        user_sims = np.sum(sims, axis=1)
        user_mean_ratings = np.mean(self.users_movies, axis=1).reshape((self.users, 1))
        ratings_diff = self.users_movies - user_mean_ratings
        pred = user_mean_ratings + (np.dot(sims, ratings_diff).T / user_sims).T
        return pred

    def recomment(self, user_index):
        predictions = self.predict()
        user = self.users_movies[user_index, :]
        pred_user = predictions[user_index, :]
        zero_idxs = np.where(user == 0)[0].flatten()
        for idx in zero_idxs:
            if pred_user[idx] >= 2:
                print(pred_user[idx], idx)


u = UserBasedCFMovies()
u.transform_ids()
u.construct_array()
u.recomment(2)
