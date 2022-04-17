import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import os

data_file_name = 'minist.npz'
if os.path.exists(data_file_name):
    data = np.load(data_file_name, allow_pickle=True)
    X = data['X']
    y = data['y']
else:
    X, y = fetch_openml('mnist_784', data_home='./', return_X_y=True)
    X = X / 255.
    np.savez(data_file_name, X=X, y=y)

# it creates mldata folder in your root project folder
# rescale the data, use the traditional train/test split
_X_train, _X_test = X[:60000], X[60000:]
_y_train, _y_test = y[:60000], y[60000:]

# %%
small_dataset = True
if small_dataset:
    X_train = _X_train[0:1000]
    y_train = _y_train[0:1000]
    X_test = _X_test[0:1000]
    y_test = _y_test[0:1000]
else:
    X_train = _X_train[:]
    y_train = _y_train[:]
    X_test = _X_test[:]
    y_test = _y_test[:]

# X_small = X[:1000]
# y_small = y[:1000]
# print(X_small.shape)

# %%
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext(conf=conf)
# %%
a = sc.parallelize(X_train)
a.count()

# %%

# apply k-means and visualize clusters' center
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=10, init='k-means++').fit(X_train)  # k-means++ is better than randomly
kmeans.labels_ = y_train
centers = np.reshape(kmeans.cluster_centers_, (10, 28, 28))

plt.figure()
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.imshow(centers[i])
plt.savefig("cluster_centers.png")
plt.show()

# visualize label histogram of one cluster
y_cluster = kmeans.predict(X_test)
y_cluster1 = []
for i in range(len(y_cluster)):
    if y_cluster[i] == 0:
        y_cluster1.append(y_test[i])

plt.hist(y_cluster1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.savefig("label_histogram_of_one_cluster.png")
plt.show()
