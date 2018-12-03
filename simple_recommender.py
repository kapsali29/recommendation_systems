"""
Simple recommenders: offer generalized recommendations to every user, based on movie popularity and/or genre. The basic idea behind this system
is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.
IMDB Top 250 is an example of this system.
"""
import pandas as pd


class SimpleRecommender(object):

    def __init__(self):
        self.movies = pd.read_csv("data/movies_metadata.csv")
        self.C = self.movies["vote_average"].mean()
        self.m = self.movies["vote_count"].quantile(q=0.9)
        self.q_movies = self.movies.copy().loc[self.movies["vote_count"] >= self.m]

    def weighted_rating(self, q_movies):
        """
        IMDB rating formula calculated
        :param q_movies: movies dataset
        :return: formula
        """
        vote_count = q_movies["vote_count"]
        vote_average = q_movies["vote_average"]
        formula = (vote_count / (vote_count + self.m)) * vote_average + self.C * (
                self.m / (vote_count + self.m))
        return formula

    def main(self):
        """
        Main function that sum up all the information
        :return:
        """
        self.q_movies["score"] = self.q_movies.apply(self.weighted_rating, axis=1)
        self.q_movies.sort_values(["score"], ascending=False)
        print(self.q_movies[["title", "score", "vote_average", "vote_count"]].head(15))


system = SimpleRecommender()
system.main()
