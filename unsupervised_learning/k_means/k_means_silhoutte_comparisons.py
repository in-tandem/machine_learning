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
from matplotlib import cm


X, y = make_blobs(

    n_samples = 400,
    n_features = 2,
    shuffle = True,
    centers = 3,
    cluster_std = 1,
    random_state = 1

)

random_k_guesses = [2, 3, 4, 5, 6, 8]

distortions = []

for random_k in random_k_guesses:

    k_means = KMeans( n_clusters = random_k, init = 'k-means++', random_state = 1 , n_init = 10, max_iter = 300)

    label_predictions = k_means.fit_predict(X)
    
    distortions.append(k_means.inertia_)
    silhouettes = silhouette_samples(X, label_predictions)
    silhouette_mean = np.mean(silhouettes)

    # print('SSE for within cluster for n %s is %s, sample silhouette coeff is %s, silhouettes are %s' %(random_k, k_means.inertia_, silhouette_mean, silhouettes))

    print("unique clusters are ", np.unique(label_predictions))
    figure , axes =  plot.subplots(1, 2) ## we will draw silhouette and scatter plot side by side

    first_plot = axes[0] ## this is our silhouette plot
    second_plot = axes[1] ## this is our scatter plot

    first_plot.set_xlim([-0.1, 1]) ## silhouette coeff can range between -1 and 1 

    y_axes_lower, y_axes_upper, y_ticks = 0 , 0 , []

    for i in range(random_k): ## unique clusters are values ranging from 0 to n_clusters used for KMeans

        #  Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = silhouettes[label_predictions == i]
        y_axes_upper = y_axes_lower + len(ith_cluster_silhouette_values)

        ith_cluster_silhouette_values.sort()

        color = cm.jet(float(i) / random_k)
    
        first_plot.fill_betweenx(np.arange(y_axes_lower, y_axes_upper), 0, ith_cluster_silhouette_values,  facecolor = color,  edgecolor=color, alpha=0.7)
        first_plot.text(-0.15, y_axes_lower + 0.5 * ith_cluster_silhouette_values.shape[0], str(i))
        
        y_axes_lower = y_axes_upper 
        

    first_plot.axvline(silhouette_mean, color = 'red', linestyle = '--')
    
    first_plot.set_yticks( [] )
    first_plot.set_title("Silhouette Plot for K %s " %random_k)
    first_plot.set_xlabel("Silhouette Coefficient")
    first_plot.set_ylabel('Cluster')

    colors = cm.jet(label_predictions.astype(float) / random_k)
    second_plot.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k') ## i am plotting what was presented, and not what was predicted

    # Labeling the clusters
    centers = k_means.cluster_centers_
    # Draw white circles at cluster centers
    second_plot.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        second_plot.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    second_plot.set_title("The visualization of the clustered data.")
    second_plot.set_xlabel("Feature space for the 1st feature")
    second_plot.set_ylabel("Feature space for the 2nd feature")
    

# plot.grid()
   
figure, ax = plot.subplots(1,1)   

ax.plot(random_k_guesses, distortions, marker= 's')
ax.set_title('Elbow Chart for K Means ')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Distortions')

plot.show()  





