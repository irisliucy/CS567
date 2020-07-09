import numpy as np
import scipy.spatial.distance as dist
from scipy.spatial import minkowski_distance
from utils import Distances

A = [1, 0, 0]
B = [0, 1, 0]



print('Scipy Euclidean distance is', dist.euclidean(A, B))
print('Euclidean distance is', Distances.euclidean_distance(A, B))
assert dist.euclidean(A, B) == Distances.euclidean_distance(A, B)

print('Scipy minkowski distance is', minkowski_distance(A, B, p=3))
print('minkowski distance is', Distances.minkowski_distance(A, B))
assert minkowski_distance(A, B, p=3) == Distances.minkowski_distance(A, B)

print('Scipy Canberra distance is', dist.canberra(A, B))
print('Canberra distance is', Distances.canberra_distance(A, B))
assert dist.canberra(A, B) == Distances.canberra_distance(A, B)

print('Scipy Cosine distance is', dist.cosine(A, B))
print('Cosine distance is', Distances.cosine_similarity_distance(A, B))
assert dist.cosine(A, B) == Distances.cosine_similarity_distance(A, B)

assert np.dot(A,B) == Distances.inner_product_distance(A,B)

print('All distance measures are the same!')