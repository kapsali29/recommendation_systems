import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


class UserBasedCFSample(object):
    def __init__(self, k, user, item):
        self.M = np.asarray([[3, 7, 4, 9, 9, 7],
                             [7, 0, 5, 3, 8, 8],
                             [7, 5, 5, 0, 8, 4],
                             [5, 6, 8, 5, 9, 8],
                             [5, 8, 8, 8, 10, 9],
                             [7, 7, 0, 4, 7, 8]])
        self.cosine_matrix = 1 - pairwise_distances(self.M, metric="cosine")
        self.k = k
        self.user = user
        self.item = item

    def find_k_neighbors(self):
        neigh = NearestNeighbors(algorithm="brute", metric="cosine")
        neigh.fit(self.M)
        user = self.M[self.user, :]
        user_reshaped = user.reshape((1, user.shape[0]))
        distances, indices = neigh.kneighbors(user_reshaped, self.k + 1)
        similarities = 1 - distances.flatten()
        indices = indices.flatten()
        idxs = []
        sims = []
        print(" The user {} k nearest neighbors are the following:".format(self.user))
        for i in range(self.k + 1):
            if indices[i] == self.user:
                continue
            else:
                idxs.append(indices[i])
                sims.append(similarities[i])
                print(" User: {} with distance {}: ".format(indices[i], similarities[i]))
        return idxs, sims

    def predict_rating(self):
        indices, similarities = self.find_k_neighbors()
        m_indices = list(range(self.M.shape[0]))
        user_ratings = self.M[self.user, :]
        similarities_sorted = np.asarray([x for _, x in sorted(zip(indices, similarities))])
        indices_to_throw = [idx for idx in m_indices if idx not in indices] + [self.user]
        neighbors_rating = np.delete(self.M, indices_to_throw, axis=0)
        mean_user_rating = np.mean(user_ratings)
        similarities_sum = np.sum(similarities_sorted)
        neighbors_item_rating = neighbors_rating[:, self.item]
        mean_neigh_ratings = np.mean(neighbors_rating, axis=1)
        formula = mean_user_rating + np.sum(((neighbors_item_rating - mean_neigh_ratings) * similarities_sorted)) / (
            similarities_sum)
        return round(formula)

    def main(self):
        print(self.predict_rating())


if __name__ == '__main__':
    u = UserBasedCFSample(k=4, user=2, item=3)
    u.main()
