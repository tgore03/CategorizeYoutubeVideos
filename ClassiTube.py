from io import open

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as col
from astor.source_repr import delimiter_groups

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

LOG=True

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

# Build a neural network model for input data, validate it and run on test data set
def use_ann(X,y):
    #Perform Neural Networks
    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=10, \
                              verbose=1, mode='auto')
    callbacks_list = [earlystop]

    # Hyper-Parameters
    momentum_rate = 0.4
    filters = 1024
    epochs = 10
    batch_size = 100
    learning_rate = 0.05
    neurons = 500
    hidden_units = 'linear'
    error_function = 'categorical_crossentropy'

    # Neural Network Model
    print("\n\n\nNeural Networks")
    if LOG:
        print("len(X[0]):", len(X[0]))
        print("len(set(y)):", len(set(y)))

    model = Sequential()
    model.add(Dense(300, input_dim=len(X[0]), activation=hidden_units))  # First hidden layer
    model.add(Dense(300, activation=hidden_units))  # Second hidden layer
    model.add(Dense(300, activation=hidden_units))  # Third hidden layer
    model.add(Dense(300, activation=hidden_units))
    model.add(Dense(300, activation=hidden_units))
    model.add(Dense(len(set(y)), activation='softmax'))  # Softmax function for output layer

    # Split dataset to train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y.T, test_size=0.20, random_state=1)

    # Stochastic Gradient Descent for Optimization
    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=momentum_rate, nesterov=True)
    adam = optimizers.Adam();

    # Compile the model
    model.compile(loss=error_function, optimizer=adam, metrics=['accuracy'])
 
    # 1-of-c output encoding
    Y_train = np_utils.to_categorical(y_train)
    print("Y_train: ",Y_train.shape)
        
    model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=100, verbose=0, callbacks=callbacks_list)

    predictions = model.predict(X_test)

    y_pred = decode_output(y_test,predictions)
    #print_statistics(y_train,y_pred)
    print_statistics(y_test,y_pred, deep=True)

    return model

def do_isomap(X, y=None):
    print("isomap")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Component 0')
    ax.set_ylabel('Component 1')
    ax.set_zlabel('Component 2')

    imap = Isomap(n_components=3,
                  n_neighbors=5,
                  n_jobs=2,
                  neighbors_algorithm='auto')
    X_n = imap.fit_transform(X)
    print("Reconstruction Error:", imap.reconstruction_error())

    if y is not None:
        colors = [float(i) for i in y]
        ax.scatter(X_n[:,0], X_n[:,1], X_n[:,2], c=colors, marker='.')
    else:
        ax.scatter(X_n[:,0], X_n[:,1], X_n[:,2], c='r', marker='.')
    plt.show()

def do_kmeans(X):
    print("\nKMeans")

    k_range=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25]
    for i in k_range:
        start_time = time.clock()
        kmeans = KMeans(n_clusters=i, n_init=20)
        y = kmeans.fit_predict(X)

        do_isomap(X, y)
        #print("Score=", kmeans.score(X))
        print("For k=%d, Silhouette Coefficient: %0.3f"
              % (i, silhouette_score(X, kmeans.labels_)))
        print("Time to build KMeans model = ", time.clock()-start_time)
    return kmeans


def do_AHC(X):
    print("TODO")
    #idx  = 1
    for n_clusters in range(2,21,1):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(('average', 'complete', 'ward')):
            plt.subplot(1, 3, index+1)
            #idx+=1
            model = AgglomerativeClustering(linkage=linkage,
                                            n_clusters=n_clusters)
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            imap = Isomap(n_components=2,n_neighbors=5,neighbors_algorithm='auto')
            X = imap.fit_transform(X)
            #print(imap.reconstruction_error())
            print("Number of Cluster=%d, linkage = %s, Silhouette Coefficient: %0.3f"% (n_clusters, linkage, silhouette_score(X, model.labels_)))
            plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
            plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
                      fontdict=dict(verticalalignment='top'))
            plt.axis('equal')
            plt.axis('off')

            plt.subplots_adjust(bottom=0, top=.89, wspace=0, left=0, right=1)
            plt.suptitle('n_cluster=%i' % (n_clusters), size=17)
        plt.savefig("num_clusters_fwd_"+str(n_clusters)+".png",bbox_inches="tight",dpi=600)
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

# Build the K-Nearest Neighbors model
def use_kNN(X,y,l,h):
    # Split dataset to train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y.T, test_size=0.20, random_state=None)
    k_range = range(l,h)
    scores = []
    for k in k_range:
        print("\n\n\nK - Nearest Neighbors")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = print_statistics(y_test, y_pred, deep=False)
        scores.append(accuracy)

    return knn, k_range, scores

# plot the kNN accuracy for different values of k
def plot_kNN(k_range, scores):
    # plot the relationship between K and testing accuracy
    # plt.plot(x_axis, y_axis)
    plt.plot(k_range, scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.show()

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

    #do_isomap(X, y)

    #KMeans
    start_time = time.clock()
    kmeans = do_kmeans(X)
    print("Time to build KMeans model = ", time.clock()-start_time)

    #Hierarchical Clustering
    start_time = time.clock()
    hc = do_AHC(X)
    print("Time to build Hierarchical Clustering model = ", time.clock()-start_time)




    # # ANN
    # start_time=time.clock()
    # ann = use_ann(X,y)
    # print("Time to build ANN model = ", time.clock()-start_time)

    # # kNN
    # kmin = 1
    # kmax = 20
    # start_time=time.clock()
    #
    # knn, k_range, scores = use_kNN(X,y,kmin,kmax)
    # print("Time to build KNN model = ", time.clock()-start_time)
    # plot_kNN(k_range, scores)
    
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



    # # use ANN
    # start_time=time.clock()
    # prediction = ann.predict(X)
    # y_test = y.T
    # y_pred = decode_output(y_test,prediction)
    # print_statistics(y_test,y_pred, deep=False)
    # print("y_test=",y_test, " y_pred=",y_pred)
    # print("Time to test ANN model = ", time.clock()-start_time)
    #
    # # use kNN
    # start_time=time.clock()
    # y_pred = knn.predict(X)
    # y_test = y.T
    # print_statistics(y_test,y_pred, deep=False)
    # print("y_test=",y_test, " y_pred=",y_pred)
    # print("Time to test KNN model = ", time.clock()-start_time)

# main method 
def main(file_train, file_test):
    # Preprocess data and store it in file
    # X, y, label_map, tfidf, pca = process_data(file_train)
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
    #tfidf, pca, ann, knn = build_models(file_train)

    # Predicting new data point using ANN and kNN
    #test_models(label_map,tfidf,pca,kmeans,hc)

# Training and testing the model on input dataset
file_train ='USvideos_modified.csv' 
file_test ='USvideos_modified_test.csv' 

main(file_train, file_test)
