import numpy as np
import tensorflow as tf
import pickle
import tabulate

def loadDataCollaborativeFiltering():
    # load ratings
    data = np.genfromtxt("data/u.data", delimiter="\t",dtype=np.int32) # shape (100000,4)
    users = data[:,0]
    movies = data[:,1]

    # will help us construct our matrix later
    user_mapping = {uid:i for i, uid in enumerate(np.unique(users))}
    movie_mapping = {mid:i for i,mid in enumerate(np.unique(movies))}

    # empty matrix - shape (num_movies,num_users)
    Y = np.zeros((len(movie_mapping), len(user_mapping)))

    # populate matrix
    for row in data:
        user_idx = user_mapping[row[0]]
        movie_idx = movie_mapping[row[1]]
        rating = row[2]
        Y[movie_idx,user_idx]=rating

    R = (Y!=0).astype(np.float32)
    return Y,R, movie_mapping

def normaliseRatings(Y):
    Ymean = np.mean(Y,axis=1,keepdims=True)
    Ynorm = Y-Ymean
    return Ynorm, Ymean

def train_cf_model(X,W,b,Y,R,costFunc,lambda_,iters):
    """
    Train Collaborative Filtering model (X,W,b) parameters on Y ratings
    """
    optimizer = tf.keras.optimizers.Adam()

    for i in range(iters):
        with tf.GradientTape() as tape:
            # forward pass
            cost = costFunc(X,W,b,Y,R,lambda_)
        
        # backward propagation for gradients
        grads = tape.gradient(cost, [X,W,b])

        # updates
        optimizer.apply_gradients(zip(grads,[X,W,b]))

        # print results
        if i%100==0:
            print(f"Training loss at iteration {i}: {cost:.2f} ")

def load_raw_features():
    """
    Load raw,unprocessed movie and user features from data files
    """
    movie_raw_features = np.genfromtxt("data/u.item", delimiter="|",dtype=str) # shape (1682,24)
    user_raw_features = np.genfromtxt("data/u.user", delimiter="|",dtype=str) # shape (943,5)

    return movie_raw_features,user_raw_features

def load_cleaned_features():
    """
    Loads cleaning movie and user features
    """
    movie_features = np.load("cleaned_data/movie_features.npy",allow_pickle=True)
    movie_features_headers = np.load("cleaned_data/movie_features_headers.npy",allow_pickle=True)
    user_features = np.load("cleaned_data/user_features.npy",allow_pickle=True)
    user_features_headers = np.load("cleaned_data/user_features_headers.npy",allow_pickle=True)

    return movie_features,movie_features_headers,user_features,user_features_headers

def load_supplementary():
    with open("cleaned_data/movie_dict.pkl","rb") as f:
        return pickle.load(f)