import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Create a scatterplot of the vector embeddings after PCA dimensionality reduction
def createPCA(dataframe_normalized, vectors, nClusters):
    # Extract the true labels from the input DataFrame
    df_teste = dataframe_normalized.copy()
    y_true = df_teste['trueLabel']

    # Scale the vectors using StandardScaler
    Sc = StandardScaler()
    X = Sc.fit_transform(vectors)

    # Apply PCA to reduce the dimensionality of the vectors to 3
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X)

    # Set the size of the plot using seaborn settings
    sns.set(rc={'figure.figsize':(13,9)})
  
    # Set the colors for the plot using a color palette
    palette = sns.color_palette("hls", n_colors=nClusters)

    # Create the scatterplot
    g = sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=y_true, palette=palette) #legend='full'

    # Move the legend outside the plot area
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Set the plot title
    plt.title('PCA')

    # Show the plot
    plt.show()

    # If we want to save the plot in png
    #plt.savefig("cluster_tsne.png")

# Create a scatterplot of the vector embeddings after t-SNE dimensionality reduction
def createTSNE(dataframe_normalized, vectors, nClusters):
    # Extract the true labels from the input DataFrame
    df_teste = dataframe_normalized.copy() 
    y_true = df_teste['trueLabel']

    # Scale the vectors using StandardScaler
    Sc = StandardScaler()
    X = Sc.fit_transform(vectors)

    # Apply t-SNE to reduce the dimensionality of the vectors to 2
    tsne = TSNE(perplexity=50, learning_rate=10, n_iter = 250)
    X_embedded = tsne.fit_transform(X)

    # Set the size of the plot using seaborn settings
    sns.set(rc={'figure.figsize':(13,9)})

    # Set the colors for the plot using a color palette
    palette = sns.color_palette("hls", n_colors=nClusters)

    # Create the scatterplot
    g = sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y_true, palette=palette)

    # Move the legend outside the plot area
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Set the plot title
    plt.title('t-SNE')

    # Show the plot
    plt.show()