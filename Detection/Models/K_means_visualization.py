# Created by: Daniela Chanci Arrubla
# Description: Application of PCA and K_means to the features extracted from each patient of the
# publicly available dataset "UPenn and Mayo Clinic's Seizure Detection Challenge" downloaded
# from kaggle

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Define path
dir = Path("F:/Users/user/Desktop/EMORY/Classes/Fall_2021/CS_534/Project/Detection/Extracted_Features/Train")
files = list(dir.glob('*.csv'))

for filename in files:

    # Load the extracted features
    data_original = pd.read_csv(filename)
    data = data_original.drop(data_original.iloc[:,[-1]], axis = 1)  # Discard  class

    # Dimensionality reduction with Principal Component Analysis
    data_reduced = PCA(n_components = 2).fit_transform(data)  # The output will have two dimensions

    # Plot PCA=2 of the data
    plt.scatter(data_reduced[:,0],data_reduced[:,1], c='black', s=10)
    plt.title("PCA on Epilectic Seizure Data", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=13)
    plt.ylabel("Principal Component 2", fontsize=13)

    # Apply k-means algorithm
    k_means = KMeans(n_clusters = 2).fit(data_reduced)
    y_kmeans = k_means.predict(data_reduced)
    labels = np.transpose(k_means.labels_)
    centers = k_means.cluster_centers_

    # Plot clustered data 2D
    fig1 = plt.figure()
    plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=y_kmeans, s=10, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=5, alpha=0.8)
    plt.title("Clustered Data", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=13)
    plt.ylabel("Principal Component 2", fontsize=13)
    plt.show()
