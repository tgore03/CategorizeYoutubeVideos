from io import open

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as col
from astor.source_repr import delimiter_groups

import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import Isomap
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import optimizers

LOG=False

# Read the input data, store it in numpy array, preprocess it and maps the labels to indexes 
def read_data(file, label_values=None):
    #Read Data
    data = pd.read_csv(file)
    if LOG:
        print("Columns in Data file:\n",data.dtypes)
        print()

    #Delete unnecessary columns
    columns=['video_id', 'last_trending_date', 'publish_date', 'publish_hour',
             'views', 'likes', 'dislikes', 'comment_count', 'comments_disabled',
             'ratings_disabled', 'tag_appeared_in_title_count', 'tag_appeared_in_title',
             'trend_day_count', 'trend.publish.diff', 'trend_tag_highest',
             'trend_tag_total','subscriber','channel_title']

    data = data.drop(columns, axis=1)
    if LOG:
        print("Remaining Columns after deletion:\n",data.dtypes,"\n")

    #Extract Labels from data and convert data to Numpy Matrix
    labels = np.array(data['category_id'].tolist())   
    data = data.as_matrix(columns = ['tags', 'title', 'description', 'tags_count'])
    #data = data.as_matrix(columns = ['tags'])
    data = data.astype(str)

    if LOG:
        print("After label data split")
        print("Shape of Data:",data.shape)
        print("Shape of Labels:",labels.shape)
        print()

    #Replace "|" in Tags with " "
    for i in range(len(data)):
        data[i,0] = data[i,0].replace('|', ' ')


    #Merge multiple columns into one column
    sep=" "
    for i in range(len(data)):
        data[i,0] = sep.join(data[i])

    #Maps the values in label to range(1, 16)
    y = np.empty(shape=labels.shape, dtype=int)
    if label_values is None:
        label_values = set(labels)
        label_values = list(label_values)

        if LOG:
            print("Uniques values in label:",label_values)
            print("No of Unique values:",len(label_values))

        for i in range(len(labels)):
            y[i] = label_values.index(labels[i])
    else:
        for i in range(len(labels)):

            y[i] = list(label_values).index(labels[i])

    if LOG:
        print("After Mapping labels to positions of its unique values")
        print("labels: ", y)
        print("unique values in labels", set(y))
        print()

    return data[:,0],y, label_values

#Store X obtained after PCA transformation
def store_array(filename, array):
    np.save(filename, array)

def store_instance(filename, instance):
    pickle.dump(instance, open(filename, 'wb'))
    if LOG:
        print(instance.get_params())

def read_array(filename):
    array= np.load(filename)
    return array

#Read PCA Transformed X from file
def read_instance(filename):
    instance = pickle.load(open(filename, 'rb'))
    if LOG:
        print(instance.get_params())
    return instance

# Compute the TF-IDF vectors for data points
def tf_idf(data):

    #TFIDF
    vectorizer = TfidfVectorizer(stop_words='english')

    X = vectorizer.fit_transform(data)
    if LOG:
        print("TFIDF vectors:")
        print("Vector shape:", X.shape)
        print()

    X = X.todense()
    if LOG:
        print("After Dense operation")
        print("Vector Shape:", X.shape)
        print()

    return X, vectorizer

# Perform the PCA on the input data set
def do_pca(X):
    #PCA

    pca = PCA(0.95)
    pca.fit(X)
    X = pca.transform(X)

    if LOG:
        print("PCA")
    #     print("Explained Variance:", pca.explained_variance_)
    #     print("Explained Variance Ratio Sum:", pca.explained_variance_ratio_.sum())
    #     print("PCA Transformed Data Shape:\n",X.shape)
    #     print()
    return X, pca

def do_isomap(X,n_comp):
    print("Performing isomap")
    imap = Isomap(n_components=n_comp,
                  n_neighbors=5,
                  n_jobs=2,
                  neighbors_algorithm='auto')
    X_n = imap.fit_transform(X)
    print("Reconstruction Error:", imap.reconstruction_error())
    return X_n

def plot2d_isomap(X, title, y=None):
    plt.title(title)
    plt.xlabel('Component 0')
    plt.ylabel('Component 1')
    if y is not None:
        colors = [float(i)% len(set(y)) for i in y]
        plt.scatter(X[:,0], X[:,1], c=colors, marker='.')
    else:
        plt.scatter(X[:,0], X[:,1], c='r', marker='.')
    #plt.show()
    plt.savefig(""+str(title)+".png", dpi = 600)

def plot_isomap(X, title, y=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('Component 0')
    ax.set_ylabel('Component 1')
    ax.set_zlabel('Component 2')
    if y is not None:
        colors = [float(i) for i in y]
        ax.scatter(X[:,0], X[:,1], X[:,2], c=colors, marker='.')
    else:
        ax.scatter(X[:,0], X[:,1], X[:,2], c='r', marker='.')
    #plt.show()
    plt.savefig(""+str(title)+".png", dpi = 600)

def do_kmeans(X, X_n=None):
    if X_n is None:
        X_n = do_isomap(X,2)

    print("\nKMeans")
    k_range=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25]
    for i in k_range:
        start_time = time.clock()
        kmeans = KMeans(n_clusters=i, n_init=20)
        y = kmeans.fit_predict(X)
        print("Time to build KMeans model = ", time.clock()-start_time)

        title="Kmeans, k-"+str(i)
        plot2d_isomap(X_n, title, y)
        #print("Score=", kmeans.score(X))
        print("For k=%d, Silhouette Coefficient: %0.3f"
              % (i, silhouette_score(X, kmeans.labels_)))
    return kmeans

def plot_isomap2d_AHC(X_n,model,index,linkage,n_clusters,elapsed_time):
    print("isomap 2D")
    plt.subplot(1, 3, index+1)
    plt.scatter(X_n[:, 0], X_n[:, 1], c=model.labels_, marker = '.')
    plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
              fontdict=dict(verticalalignment='top'))
    plt.axis('equal')
    plt.axis('off')
    plt.subplots_adjust(bottom=0, top=.89, wspace=0, left=0, right=1)
    plt.suptitle('n_cluster=%i' % (n_clusters), size=17)

def plot_isomap3d_AHC(X_n,model,index,linkage,n_clusters,elapsed_time):
    print("isomap 3D")
    plt.subplot(111, projection='3d')
    plt.scatter(X_n[:, 0], X_n[:, 1], X_n[:, 2], c=model.labels_, marker = '.')
    plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
              fontdict=dict(verticalalignment='top'))
    plt.axis('equal')
    plt.axis('off')
    plt.subplots_adjust(bottom=0, top=.89, wspace=0, left=0, right=1)
    plt.suptitle('n_cluster=%i' % (n_clusters), size=17)
    plt.savefig("3D_num_clusters_"+str(linkage)+str(n_clusters)+".png",bbox_inches="tight",dpi=600)

def silhouette_plot(X, l, h):
    for n_clusters in range(l,h):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = AgglomerativeClustering(linkage='complete',n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
     
     # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.Spectral(float(i) / n_clusters)
            #colors = [float(i) for i in cluster_labels]
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = plt.cm.Spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        #centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        #ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
        #            c="white", alpha=1, s=200, edgecolor='k')

        #for i, c in enumerate(centers):
        #    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
        #                s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.show()

def do_AHC(X):
    print("Aglomerative Hierarchical clustering")
    for n_clusters in range(5,20):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(('average', 'complete', 'ward')):
            model = AgglomerativeClustering(linkage=linkage,
                                            n_clusters=n_clusters)
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            
            print("Number of Cluster=%d, linkage = %10s, Silhouette Coefficient: %0.3f"% (n_clusters, linkage, silhouette_score(X, model.labels_)))
            
            #plot 2d
            X_n = do_isomap(X,2)
            plot_isomap2d_AHC(X_n,model,index,linkage,n_clusters,elapsed_time)
        
            #plot 3d
            #X_n = do_isomap(X,3)
            #plot_isomap3d_AHC(X_n,model,index,linkage,n_clusters,elapsed_time)

        plt.savefig("num_clusters_"+str(n_clusters)+".png",bbox_inches="tight",dpi=600)
    plt.show()
    return None

# decode c dimension output to 1 dimension
def decode_output(y_test,predictions):
    # 1-of-c output decoding
    y_pred = np.empty(shape = y_test.shape)
    i=0
    for row in predictions:
        y_pred[i] = np.argmax(row)
        i+=1
    return y_pred

# print the confusion matrix, class accuracies and overall accuracy
def print_statistics(y_test,y_pred, deep=False):
    accuracy = accuracy_score(y_test, y_pred)

    if deep:
        matrix = confusion_matrix(y_test, y_pred)
        print(len(matrix))
        sum = 0
        print("Class Accuracies:")
        for i in range(len(matrix)):
            sum += matrix[i][i]
            print("Class ", i, ": ", round(matrix[i][i]/np.sum(matrix[i]), 4))
        print("Confusion Matrix:\n", matrix)
    print("Overall Accuracy:\n", accuracy)
    return accuracy

# Preprocess data and store it in the file
def process_data(f):
    data, y, label_map = read_data(f)
    X, tfidf = tf_idf(data)
    X, pca = do_pca(X)
    np.savetxt("X.csv", X, delimiter=",")
    np.savetxt("y.csv", y, delimiter=",")
    
#Store the result
    store_array("X_array", X)
    store_array("y_array", y)
    store_array("label_map",label_map)
    store_instance("tfidf.sav",tfidf)
    store_instance("pca.sav", pca)
    return X, y, label_map, tfidf, pca

# method to build ANN and kNN models

def build_models(f, X=None, y=None, label_map=None, tfidf=None, pca=None):

    if X is None:
        data, y, label_map = read_data(f)
        X, tfidf = tf_idf(data)
        X, pca = do_pca(X)

    #do_isomap(X)

    #KMeans
    start_time = time.clock()
    kmeans = 0#do_kmeans(X)
    print("Time to build KMeans model = ", time.clock()-start_time)

    #Hierarchical Clustering
    start_time = time.clock()
    hc = 0#do_AHC(X)
    silhouette_plot(X,20,21)
    print("Time to build Hierarchical Clustering model = ", time.clock()-start_time)

    return label_map, tfidf, pca, kmeans, hc

# Test the new data point on already build models
def test_models(label_map, tfidf, pca, kmeans, hc):

    print("\n\n\nTesting the data")

    data, y, label_map = read_data(file_test, label_map)
    #X = tfidf.fit_transform(data[:,2])
    X = tfidf.transform(data)
    X = X.todense()

    X = pca.transform(X)

    #KMeans
    start_time = time.clock()
    y_pred = kmeans.predict(X)
    print("y_pred=", y_pred)
    print("Predicted Class = ", y_pred)
    y_test = y.T
    print_statistics(y_test,y_pred, deep=False)
# main method 
def main(file_train, file_test):
    # Preprocess data and store it in file
    #X, y, label_map, tfidf, pca = process_data(file_train)
    #print("Data Processed")

    #Read data from files
    X=read_array("X_array.npy")
    y=read_array("y_array.npy")
    label_map=read_array("label_map.npy")
    tfidf=read_instance("tfidf.sav")
    pca=read_instance("pca.sav")
    print("Data read")

    # Buildling ANN and kNN models
    label_map, tfidf, pca, kmeans, hc = build_models(file_train, X, y, label_map, tfidf, pca)

    # Predicting new data point using ANN and kNN
    #test_models(label_map,tfidf,pca,kmeans,hc)

# Training and testing the model on input dataset
file_train ='USvideos_modified.csv' 
file_test ='USvideos_modified_test.csv' 

main(file_train, file_test)
