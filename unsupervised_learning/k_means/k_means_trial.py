from matplotlib import pyplot as plot 
import pandas as panda 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

X, y = make_blobs(

    n_samples = 10,
    n_features = 2,
    centers = 3,
    shuffle = True, ## if False, ordered by labels,eg all 0s appear first, then all 1 s, etc. if True, not ordered
    cluster_std = 0.85,
    random_state = 2
    
)

plot.scatter(X[:, 0], X[:, 1], c ='red', edgecolors='black', s = 90, marker = 'o')
plot.grid()
# plot.show()

k_means_usual = KMeans(init = 'random', n_clusters = 3, n_init = 10, max_iter = 300, random_state = 2 )


label_predictions = k_means_usual.fit_predict(X) ## equivalent to calling fit(X) and then predict(X)

print(k_means_usual.labels_) ## labels of each point
print(k_means_usual.cluster_centers_ ) ## ndarray as the name suggests, lenght is no of clusters,
# and size of each is the sixe of features. so 3*2 for 3 clusters 2d, 3*3 for 3 clusters, 3 features
print(k_means_usual.inertia_ ) ## SSE for within clusters  
#ie sum of sqaured distance error of each sample to centre of cluster. can be used for plotting elbow

