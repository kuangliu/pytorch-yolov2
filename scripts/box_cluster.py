'''Cluster boxes to K centroids with K-means.

Note:
  The actual anchor boxes used is from the Darknet config file: yolo-voc.cfg
  anchors = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]

What I get:
  anchors = [(0.6240, 1.2133), (1.4300, 2.2075), (2.2360, 4.3081), (4.3940, 6.5976), (9.5680, 9.9493)]
'''
import torch
import random

from utils import iou


def init_centroids(X, K):
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
        ious = iou(centroids, X[torch.LongTensor(other_ids)])
        min_ious, _ = ious.min(0)
        _, idx = min_ious.min(1)
        picked_ids.append(idx[0][0])
    return X[torch.LongTensor(picked_ids)]

def pick_centroid_from_cluster(X):
    '''Instead of choosing the mean of cluster as the centroid,
    I pick the centroid as a sample from the cluster with the maximum average iou.

    Args:
      X: (tensor) samples of a cluster.

    Return:
      (tensor) picked centroid from the cluster.
    '''
    best_iou = -1
    for x in X:
        iou_x = iou(x.view(1,4), X)
        if iou_x.mean() > best_iou:
            best_iou = iou_x.mean()
            centroid = x
    return centroid

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
    ids = torch.randperm(N)[:K]
    centroids = X[ids].clone()

    # Pick centroids with K-means++
    # centroids = init_centroids(X, K)

    last_loss = 0
    for it in range(max_iter):
        # Assign each sample to the nearest centroid
        groups = [[] for i in range(K)]
        dist_sum = 0
        for i in range(N):
            x = X[i].view(1,4)
            dists = 1 - iou(x, centroids)
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
            centroids.append(pick_centroid_from_cluster(X[group_i]))
        centroids = torch.stack(centroids)

        if abs(last_loss - loss) < tol:
            break
        last_loss = loss
    return centroids


random.seed(0)
torch.manual_seed(0)

K = 5  # 5 centroids
list_file = './voc_data/voc07_train.txt'
fmsize = 13

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
centroids = kmeans(boxes, K)
print(centroids * fmsize)
