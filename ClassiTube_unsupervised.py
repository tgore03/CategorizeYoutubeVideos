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

def read_data(file, label_values=None):
    '''
    Read data from file, extract label and features, convert to numpy array and hashes the category labels with its index
    :param file: file to read data from
    :param label_values:  hash map of labels with their index
    :return: data, labels, hashmap of labels
    '''
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

def store_array(filename, array):
    '''
    Store the array in file
    :param filename: file to store array in
    :param array: array to store
    :return: None
    '''
    np.save(filename, array)

def store_instance(filename, instance):
    '''
    Store an object in file
    :param filename: file to store object in
    :param instance: object to store
    :return: None
    '''
    pickle.dump(instance, open(filename, 'wb'))
    if LOG:
        print(instance.get_params())

def read_array(filename):
    '''
    Read a numpy array from file
    :param filename: file to read array from
    :return: numpy array
    '''
    array= np.load(filename)
    return array

def read_instance(filename):
    '''
    Read object stored in file
    :param filename: file to read object from.
    :return: object
    '''
    instance = pickle.load(open(filename, 'rb'))
    if LOG:
        print(instance.get_params())
    return instance

def tf_idf(data):
    '''
    converts the textual feature to numberical format
    :param data: textual features
    :return: numerical features, TFIDF model
    '''
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

def do_pca(X):
    '''
    Performs PCA dimensionality reduction on TFIDF vectors
    :param X: TFIDF matrix
    :return: PCA matrix
    '''

    pca = PCA(0.95)
    pca.fit(X)
    X = pca.transform(X)

    if LOG:
        print("PCA")
        print("Explained Variance:", pca.explained_variance_)
        print("Explained Variance Ratio Sum:", pca.explained_variance_ratio_.sum())
        print()
    return X, pca

def do_isomap(X,n_comp):
    '''
    Reduces the dimensions of data to 2 or 3.
    :param X: data
    :param n_comp: number of dimensions to reduce to
    :return: Dimension reduced data
    '''
    print("Performing isomap")
    imap = Isomap(n_components=n_comp,
                  n_neighbors=5,
                  n_jobs=2,
                  neighbors_algorithm='auto')
    X_n = imap.fit_transform(X)
    print("Reconstruction Error:", imap.reconstruction_error())
    return X_n

def plot2d_isomap(X, title, y=None):
    '''
    Plot the dimension reduced data in 2D scatter plot
    :param X: 2D data
    :param title: title for the plot
    :param y: labels used to color code the scatter plot
    :return: None
    '''
    plt.title(title)
    plt.xlabel('Component 0')
    plt.ylabel('Component 1')
    plt.legend()
    if y is not None:
        colors = [float(i)% len(set(y)) for i in y]
        plt.scatter(X[:,0], X[:,1], c=colors, marker='.')
    else:
        plt.scatter(X[:,0], X[:,1], c='r', marker='.')
    #plt.show()
    plt.savefig(""+str(title)+".png", dpi = 600)

def plot_isomap(X, title, y=None):
    '''
    Plot the dimension reduced data in 3D scatter plot
    :param X: 3D data
    :param title: title for the plot
    :param y: labels used to color code the scatter plot
    :return: None
    '''
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
    '''
    Build the KMean model, train it and print statistics.
    :param X: data to train the model on
    :param X_n: isomap data used for plotting the clusters
    :return: KMeans model
    '''
    if X_n is None:
        X_n = do_isomap(X,2)

    print("\nKMeans")
    #k_range=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25]
    k_range=[3, 4]
    for i in k_range:
        start_time = time.clock()
        kmeans = KMeans(n_clusters=i, n_init=20, n_jobs=3)
        y = kmeans.fit_predict(X)
        print("Time to build KMeans model = ", time.clock()-start_time)

        title="Kmeans, k-"+str(i)
        plot2d_isomap(X_n, title, y)
        print("For k=%d, Silhouette Coefficient: %0.3f"
              % (i, silhouette_score(X, kmeans.labels_)))
    return kmeans

def plot_isomap2d_AHC(X_n,model,index,linkage,n_clusters,elapsed_time):
    '''
    Plot a 2D scatter plot on isomap data using predicted labels obtained from AHC
    :param X_n: 2D data
    :param model: AHC model
    :param index: hashmap for labels
    :param linkage: linkage type for AHC
    :param n_clusters: No of clusters for AHC
    :param elapsed_time: Running time for training the model
    :return: None
    '''
    print("isomap 2D plot")
    plt.subplot(1, 3, index+1)
    plt.scatter(X_n[:, 0], X_n[:, 1], c=model.labels_, marker = '.')
    plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
              fontdict=dict(verticalalignment='top'))
    plt.axis('equal')
    plt.axis('off')
    plt.subplots_adjust(bottom=0, top=.89, wspace=0, left=0, right=1)
    plt.suptitle('n_cluster=%i' % (n_clusters), size=17)

def plot_isomap3d_AHC(X_n,model,index,linkage,n_clusters,elapsed_time):
    '''
    plot 3D scatter plot on isomap data using labels obtained from AHC
    :param X_n: 2D isomap data
    :param model: AHC model
    :param index: hash map for labels
    :param linkage: linkage type used
    :param n_clusters: no. of clusters used
    :param elapsed_time: Training time for the model
    :return: None
    '''
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

def do_AHC(X, l, h, X_n=None):
    '''
    Build the Agglomerative Hierarchical Clustering model, Train the model and plot the result
    :param X: data
    :param l: lower limit for number of clusters
    :param h; higher limit for number of clusters
    :return: AHC model
    '''
    print("Aglomerative Hierarchical clustering")
    if X_n is None:
        X_n = do_isomap(X,2)

    model =None
    for n_clusters in range(l,h):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(('average', 'complete', 'ward')):
            model = AgglomerativeClustering(linkage=linkage,
                                            n_clusters=n_clusters)
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            
            print("Number of Cluster=%d, linkage = %10s, Silhouette Coefficient: %0.3f"% (n_clusters, linkage, silhouette_score(X, model.labels_)))
            
            #plot 2d
            plot_isomap2d_AHC(X_n,model,index,linkage,n_clusters,elapsed_time)
        
            #plot 3d
            #X_n = do_isomap(X,3)
            #plot_isomap3d_AHC(X_n,model,index,linkage,n_clusters,elapsed_time)

        plt.savefig("num_clusters_"+str(n_clusters)+".png",bbox_inches="tight",dpi=600)
    plt.show()
    return model

def process_data(f):
    '''
    Method that handles all the pre-processing steps. Loading data, TFIDF, PCA and storing the preprocessed data
    :param f: file to read data from
    :return:Data, labels, hashmap for labels, TFIDF model, PCA model
    '''
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

def build_models(f, X=None, y=None, label_map=None, tfidf=None, pca=None):
    '''
    Method that handles the training phase of all the models.
    Performs KMeans and AHC on the dataset read from the training file.
    :param f: file to read data from if initialization failed
    :param X: data if initialized beforehand
    :param y: labels if initialized beforehand
    :param label_map: hashmap for labels if initialized beforehand
    :param tfidf: TFIDF Model if initialized beforehand
    :param pca: PCA Model if initialized beforehand
    :return: hashmap for labels, TFIDF model, PCA model, KMeans model, AHC Model
    '''
    if X is None:
        data, y, label_map = read_data(f)
        X, tfidf = tf_idf(data)
        X, pca = do_pca(X)

    n_comp = 2
    X_n = do_isomap(X, n_comp)

    #KMeans
    start_time = time.clock()
    kmeans = do_kmeans(X, X_n)
    print("Time to build KMeans model = ", time.clock()-start_time)

    #Hierarchical Clustering
    start_time = time.clock()
    hc = do_AHC(X, 2, 4, X_n)
    print("Time to build Hierarchical Clustering model = ", time.clock()-start_time)

    return label_map, tfidf, pca, kmeans, hc

def main(file_train):
    '''
    Main Method which handles the flow of the program
    :param file_train: file containing training data
    :param file_test: file contatining testing data
    :return: None
    '''

    #Preprocess data and store it in file
    X, y, label_map, tfidf, pca = process_data(file_train)
    print("Data Processed")

    #Read data from files
    X=read_array("X_array.npy")
    y=read_array("y_array.npy")
    label_map=read_array("label_map.npy")
    tfidf=read_instance("tfidf.sav")
    pca=read_instance("pca.sav")
    print("Data read")

    # Building K-Means and Agglomerative Hierarchical Clustering(AHC) models
    label_map, tfidf, pca, kmeans, hc = build_models(file_train, X, y, label_map, tfidf, pca)

file_train ='USvideos_modified.csv'

if __name__ == '__main__':
    main(file_train)
