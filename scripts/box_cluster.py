'''Cluster boxes to K centroids with K-means.'''
import torch
import random

from utils import iou


def pick_centroids(X, K):
    '''Pick K centroids with K-means++ algorithm.

    Args:
      X: (tensor) data, sized [N,D].
      K: (int) number of clusters.

    ReturnsL
      (tensor) picked centroids, sized [K,D].
    '''
    N, D = X.size()

    # Randomly pick the first centroid
    picked_ids = []
    picked_ids.append(random.randint(0, N-1))
    while len(picked_ids) < K:
        # Compute the dists to other points
        centroids = X[torch.LongTensor(picked_ids)]
        other_ids = [i for i in range(N) if i not in picked_ids]
        dists = 1 - iou(centroids, X[torch.LongTensor(other_ids)], 'xywh')
        min_dists, _ = dists.min(0)
        _, idx = min_dists.max(1)
        picked_ids.append(idx[0][0])
    return X[torch.LongTensor(picked_ids)]

def kmeans(X, K, max_iter=100, tol=1e-7):
    '''Run K-means on data X.

    Args:
      X: (tensor) data, sized [N,D].
      K: (int) number of clusters.
      max_iter: (int) max number of iterations.
      tol: (float) loss tolerance between two iterations.

    Returns:
      (tensor) centroids, sized [K,D].
    '''
    N, D = X.size()
    assert N >= K, 'Too few samples for K-means'

    # Randomly pick the centroids
    # ids = torch.randperm(N)[:K]
    # centroids = X[ids].clone()

    # Pick centroids with K-means++
    centroids = pick_centroids(X, K)

    last_loss = 0
    for it in range(max_iter):
        # Assign each sample to the nearest centroid
        groups = [[] for i in range(K)]
        dist_sum = 0
        for i in range(N):
            x = X[i].view(1,4)
            dists = 1 - iou(x, centroids, 'xywh')
            # dists = (x.expand_as(centroids) - centroids).pow(2).sum(1).sqrt()
            min_dist, centroid_idx = dists.squeeze().min(0)
            groups[centroid_idx[0]].append(i)
            dist_sum += min_dist[0]
        loss = dist_sum / N
        print('iter: %d/%d  loss: %f  avg_iou: %f' % (it, max_iter, loss, 1-loss))

        # Compute the new centroids
        centroids = []
        for i in range(K):
            group_i = torch.LongTensor(groups[i])
            centroids.append(X[group_i].mean(0))
        centroids = torch.cat(centroids, 0)

        if abs(last_loss - loss) < tol:
            break
        last_loss = loss
    return centroids


K = 5  # 5 centroids
grid_size = 13
list_file = './voc_data/voc07_train.txt'

boxes = []
f = open(list_file, 'r')
for line in f.readlines():
    splited = line.strip().split()
    img_width = float(splited[1])
    img_height = float(splited[2])
    num_boxes = (len(splited) - 3) // 5
    for i in range(num_boxes):
        xmin = float(splited[3+5*i]) / img_width
        ymin = float(splited[4+5*i]) / img_height
        xmax = float(splited[5+5*i]) / img_width
        ymax = float(splited[6+5*i]) / img_height
        w = xmax - xmin
        h = ymax - ymin
        boxes.append([0,0,w,h])

boxes = torch.Tensor(boxes)
centroids = kmeans(boxes, 5)
print(centroids * grid_size)
