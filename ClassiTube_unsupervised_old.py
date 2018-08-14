from io import open
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import Isomap
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pickle


def read_data(file, label_values=None):
    #Read Data
    data = pd.read_csv(file)

    #Delete unnecessary columns
    columns=['video_id', 'last_trending_date', 'publish_date', 'publish_hour',
             'views', 'likes', 'dislikes', 'comment_count', 'comments_disabled',
             'ratings_disabled', 'tag_appeared_in_title_count', 'tag_appeared_in_title',
             'trend_day_count', 'trend.publish.diff', 'trend_tag_highest',
             'trend_tag_total','subscriber','channel_title']
    data = data.drop(columns, axis=1)

    #Extract Labels from data and convert data to Numpy Matrix
    labels = np.array(data['category_id'].tolist())   
    data = data.as_matrix(columns = ['tags', 'title', 'description', 'tags_count'])
    #data = data.as_matrix(columns = ['tags'])
    data = data.astype(str)

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

        for i in range(len(labels)):
            y[i] = label_values.index(labels[i])
    else:
        for i in range(len(labels)):
            y[i] = list(label_values).index(labels[i])

    return data[:,0],y, label_values

def store_array(filename, array):
    np.save(filename, array)

def store_instance(filename, instance):
    pickle.dump(instance, open(filename, 'wb'))

def read_array(filename):
    array= np.load(filename)
    return array

def read_instance(filename):
    instance = pickle.load(open(filename, 'rb'))
    return instance

def tf_idf(data):
    #TFIDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    X = X.todense()
    return X, vectorizer

def do_pca(X):
    pca = PCA(0.95)
    pca.fit(X)
    X = pca.transform(X)
    return X, pca

def do_isomap(X,n_comp):
    print("Performing isomap")
    imap = Isomap(n_components=n_comp,
                  n_neighbors=5,
                  n_jobs=2,
                  neighbors_algorithm='auto')
    X_n = imap.fit_transform(X)
    return X_n

def plot2d_isomap(X, title, y=None):
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

def do_kmeans(X, X_n=None):
    if X_n is None:
        X_n = do_isomap(X,2)

    print("\nKMeans")
    k_range=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25]
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
    print("isomap 2D plot")
    plt.subplot(1, 3, index+1)
    plt.scatter(X_n[:, 0], X_n[:, 1], c=model.labels_, marker = '.')
    plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
              fontdict=dict(verticalalignment='top'))
    plt.axis('equal')
    plt.axis('off')
    plt.subplots_adjust(bottom=0, top=.89, wspace=0, left=0, right=1)
    plt.suptitle('n_cluster=%i' % (n_clusters), size=17)

def do_AHC(X, l, h, X_n=None):
    print("Aglomerative Hierarchical clustering")
    if X_n is None:
        X_n = do_isomap(X,2)

    model =None
    for n_clusters in range(l,h):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(('average', 'complete', 'ward')):
            model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            print("Number of Cluster=%d, linkage = %10s, Silhouette Coefficient: %0.3f"% (n_clusters, linkage, silhouette_score(X, model.labels_)))
            
            #plot 2d
            plot_isomap2d_AHC(X_n,model,index,linkage,n_clusters,elapsed_time)
        plt.savefig("num_clusters_"+str(n_clusters)+".png",bbox_inches="tight",dpi=600)
    plt.show()
    return model

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

def build_models(f, X=None, y=None, label_map=None, tfidf=None, pca=None):
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
