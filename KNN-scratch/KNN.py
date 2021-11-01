from collections import Counter
import math


def knn(k, data, query, dist_func, choice_func):
    neighbor_dist_indices = []

    for index, example in enumerate(data):
        distance = dist_func(example[:-1], query)
        neighbor_dist_indices.append((distance, index))

    sorted_neighbor_distances_indices = sorted(neighbor_dist_indices)
    k_nearest_distances_indices = sorted_neighbor_distances_indices[:k]

    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_indices]

    return k_nearest_distances_indices, choice_func(k_nearest_labels)


def mean(labels):
    return sum(labels) / len(labels)


def mode(labels):
    return Counter(labels).most_common(1)[0][0]


def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)


def main():

    clf_data = [
        [22, 1],
        [23, 1],
        [21, 1],
        [18, 1],
        [19, 1],
        [25, 0],
        [27, 0],
        [29, 0],
        [31, 0],
        [45, 0],
    ]
    # Question:
    # Given the data we have, does a 33 year old like pineapples on their pizza?
    clf_query = [33]
    clf_k_nearest_neighbors, clf_prediction = knn(
        3, clf_data, clf_query, dist_func=euclidean_distance, choice_func=mode
    )
    print("classification prediction: ", clf_prediction)

if __name__ == '__main__':
    main()