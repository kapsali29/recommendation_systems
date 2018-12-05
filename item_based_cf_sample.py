import numpy as np
from sklearn.neighbors import NearestNeighbors


class ItemBasedCFSample(object):
    def __init__(self, k, item_index, user_index):
        self.M = np.asarray([[3, 7, 4, 9, 9, 7],
                             [7, 0, 5, 3, 8, 8],
                             [7, 5, 5, 0, 8, 4],
                             [5, 6, 8, 5, 9, 8],
                             [5, 8, 8, 8, 10, 9],
                             [7, 7, 0, 4, 7, 8]])
        self.items = self.M.T
        self.k = k + 1
        self.item_index = item_index
        self.user_index = user_index

    def find_k_closest_items(self):
        item = self.items[self.item_index, :].reshape((1, self.items.shape[0]))
        neigh = NearestNeighbors(algorithm="brute", metric="cosine")
        neigh.fit(self.items)
        distances, indices = neigh.kneighbors(item, self.k)
        similarities = 1 - distances.flatten()
        indices = indices.flatten()
        sims = []
        idxs = []
        for i in range(len(indices)):
            if indices[i] == self.item_index:
                continue
            else:
                print(" Item {} is similar with measure {}".format(indices[i], similarities[i]))
                sims.append(similarities[i])
                idxs.append(indices[i])
        return idxs, sims

    def predict_rating(self):
        idxs, sims = self.find_k_closest_items()
        item_indices = list(range(self.items.shape[0]))
        remaining_idxs = [idx for idx in item_indices if idx not in idxs]
        neighbor_matrix = np.delete(self.items, remaining_idxs, axis=0)
        sum_sims = np.sum(sims)
        sorted_sims = np.asarray([x for _, x in sorted(zip(idxs, sims))])
        item_ratings = neighbor_matrix[:, self.user_index]
        formula = np.dot(item_ratings, sorted_sims) / sum_sims
        return round(formula)

    def main(self):
        print(self.predict_rating())


i = ItemBasedCFSample(k=2, item_index=1, user_index=1)
i.main()
