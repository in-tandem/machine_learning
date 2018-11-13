'''
We will attempt to chose a better K based on silhouette plot analysis
We are also taking a 2D blob since for this trial we would also like 
to plot the cluster divisions as well as the silhouette plot

'''

from sklearn.datasets import make_blobs
from matplotlib import pyplot as plot 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

X, y = make_blobs(

    n_samples = 200,
    n_features = 2,
    shuffle = True,
    centers = 3,
    cluster_std = 0.85,
    random_state = 2 

)

random_k_guesses = [2, 4, 5 , 6 , 7]

distortions = []

for random_k in random_k_guesses:

    k_means = KMeans( n_clusters = random_k, init = 'k-means++', random_state = 2 , n_init = 10, max_iter = 300)

    label_predictions = k_means.fit_predict(X)
    
    distortions.append(k_means.inertia_)
    silhouettes = silhouette_samples(X, label_predictions)
    silhouette_mean = np.mean(silhouettes)

    print('SSE for within cluster for n %s is %s, sample silhouette coeff is %s ' %(random_k, k_means.inertia_, silhouette_mean))



plot.plot(random_k_guesses, distortions, marker= 's')
plot.title('Elbow Chart for K Means ')
plot.xlabel('Number of Clusters')
plot.ylabel('Distortions')
plot.grid()
plot.show()






