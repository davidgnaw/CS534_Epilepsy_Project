# Created by: Daniela Chanci Arrubla
# Description: Application of PCA to the features extracted from each patient of the
# publicly available dataset "UPenn and Mayo Clinic's Seizure Detection Challenge" downloaded
# from kaggle

import pandas as pd
import os
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define path
dir = Path("F:/Users/user/Desktop/EMORY/Classes/Fall_2021/CS_534/Project/Detection/Extracted_Features/Train")
files = list(dir.glob('*.csv'))

for filename in files:

    # Load the extracted features
    data_original = pd.read_csv(filename)
    target = data_original['label']
    data_original['label'].replace(0, 'Interictal',inplace=True)
    data_original['label'].replace(1, 'Ictal',inplace=True)
    data = data_original.drop(data_original.iloc[:,[-1]], axis = 1)  # Discard  class
    data = StandardScaler().fit_transform(data) # normalize features
    labels = data_original['label']

    # Define subject name
    subject = str(filename).split("\\")[-1].split("_")[0] + "_" + str(filename).split("\\")[-1].split("_")[1]

    # Dimensionality reduction with Principal Component Analysis
    principal_components = PCA(n_components=2).fit_transform(data)  # The output will have two dimensions
    principal_df = pd.DataFrame(data=principal_components, columns=['Principal_Component_1', 'Principal_Component_2'])
    pca_df = pd.concat([principal_df, labels], axis=1)

    # Plot PCA=2 of the data
    targets = ['Interictal', 'Ictal']
    colors = ['coral', 'lightblue']

    for target, color in zip(targets,colors):
        indicesToKeep = pca_df['label'] == target
        plt.scatter(pca_df.loc[indicesToKeep, 'Principal_Component_1'], pca_df.loc[indicesToKeep, 'Principal_Component_2'],
                   c=color, edgecolors='k', s=30)
    plt.title(subject + " Detection Data", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=13)
    plt.ylabel("Principal Component 2", fontsize=13)
    plt.legend(targets)

    # Save Plot
    fig_dir = fig_dir = os.path.join("F:/Users/user/Desktop/EMORY/Classes/Fall_2021/CS_534/Project/Detection/Figures_PCA", subject+".png")
    plt.savefig(fig_dir)
    # plt.show()

