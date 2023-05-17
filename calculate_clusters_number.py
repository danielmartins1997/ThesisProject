import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# The "silhouette_method" function uses the silhouette method to find the optimal number of clusters in a k-means clustering analysis.
def silhouette_method(data, min_k, max_k, incr):
    number_clusters = 0
    atual_silhouette = 0.000
    # Prepare the scaler
    scale = StandardScaler().fit(data)

    # Fit the scaler
    scaled_data = pd.DataFrame(scale.fit_transform(data))

    # For the silhouette method k needs to start from 2
    n_clusters_axis = range(min_k, max_k, incr)
    silhouettes = []

    # Fit the method
    for k in n_clusters_axis:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init="random")
        kmeans.fit(scaled_data)
        score = metrics.silhouette_score(scaled_data, kmeans.labels_)
        silhouettes.append(score)
        #print(score)
        if score > atual_silhouette:
            number_clusters = k
            atual_silhouette = score
    
    # Plot the results
    #plt.figure(figsize=(15, 5))
    #plt.plot(n_clusters_axis, silhouettes, "bx-")
    #plt.xlabel("Values of K")
    #plt.ylabel("Silhouette score")
    #plt.title("Silhouette Method")
    #plt.grid(True)
    #plt.show()

    return number_clusters

# This is a supervised way to find the number of clusters
def choose_exact_number_labels(normalized_df):

    number_exact = normalized_df.trueLabel.nunique()
    n_clusters = number_exact


    return n_clusters