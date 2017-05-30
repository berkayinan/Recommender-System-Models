import numpy as np
import scipy

def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings=10, p_test=0.1,clean=False):
    """Split data into test and train
       If you set clean to True, it removes the users and items that has a low number of ratings.
       By default, it take 10% of data as test.
       Returns:
           valid_ratings,train,test
    """
    # set seed
    np.random.seed(988)
    if(clean==True):
        valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
        valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
        valid_ratings = ratings[valid_items, :][: , valid_users]  
        print(valid_ratings.nnz)
    else:
        valid_ratings=ratings
    rand_vector=np.random.uniform(0,1,valid_ratings.size)
    mask_train=(valid_ratings.nonzero()[0][np.where(rand_vector>=p_test)],valid_ratings.nonzero()[1][np.where(rand_vector>=p_test)])
    mask_test=(valid_ratings.nonzero()[0][np.where(rand_vector<p_test)],valid_ratings.nonzero()[1][np.where(rand_vector<p_test)])
    train=scipy.sparse.lil_matrix(valid_ratings.shape)    
    test=scipy.sparse.lil_matrix(valid_ratings.shape)
    train[mask_train]=valid_ratings[mask_train]
    test[mask_test]=valid_ratings[mask_test]
    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test


def split_data_wrapper(ratings):
    """
    A wrapper method for simple split data call
    Returns:
        ratings,train,test matrices
    """
    return split_data(ratings, 0 , 0)

def build_k_indices(rows, k_fold=4, seed=42):
    """build k indices for k-fold."""
    
    interval = int(rows / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(rows)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def split_K_fold(data,k,k_indices):
    """Split data into training and testing data
        Args:
            data: Data matrix
            k: index of test data set
            k_indices:array of random permutation k_indices
    """
    test_idx=k_indices[k]
    train_idx=np.vstack([k_indices[:k],k_indices[(k+1):]]).flatten()
    mask_train=(data.nonzero()[0][train_idx],data.nonzero()[1][train_idx])
    mask_test=(data.nonzero()[0][test_idx],data.nonzero()[1][test_idx])
    train=scipy.sparse.lil_matrix(data.shape)    
    test=scipy.sparse.lil_matrix(data.shape)
    train[mask_train]=data[mask_train]
    test[mask_test]=data[mask_test]
    return train,test


def init_MF(train, num_features):
    """Returns random matrices for user_features(NxK) and item_features(DxK)
       Args:
           train: Training matrix
           num_features: Number of features
       Returns:
           user_features,item_features
    """
    user_features=np.random.rand(train.shape[1],num_features)
    item_features=np.random.rand(train.shape[0],num_features)
    return user_features,item_features