"""
Trainer methods for respective models
Each of them returns a model object as described in models.py
"""


from modules.helpers import build_index_groups
from modules.models import *
from modules.utils import *
import time


def train_SVD_Bias(train,test,lambda_user = 40,lambda_item = 50,lambda_bias=0,
                   num_features=70,num_epochs = 10,disable_mean=False):
    """Returns a model trained by ALS, factoring user and item bias
        Returns:
            model: SVD_Bias_LS_Model
    """
    np.random.seed(988)

    # Setup model
    user_features, item_features = init_MF(train, num_features)
    bias_item = np.zeros(train.shape[0])
    bias_user = np.zeros(train.shape[1])
    if disable_mean:
        mean = 0
    else:
        mean=train.sum()/train.nnz
    model=SVD_Bias_LS_Model(item_features,user_features,bias_item,bias_user,mean)
    model.set_nz(train)
    total_start=time.time()
    for it in range(num_epochs):#Run multiple iterations
        start_time=time.time()
        print("Iteration: ",it)
        model.update_user_features(train,lambda_user,lambda_bias)#update user features
        model.update_item_features(train,lambda_item,lambda_bias)#update item features
        rmse_train=model.compute_error(train)
        print("Training RMSE",rmse_train)
        print("This iteration ran in ", time.time()-start_time , "sec")
    if test is not None:
        rmse_test=model.compute_error(test)
        print("TEST RMSE: ",rmse_test)
    print("TOTAL ELAPSED TIME:",-total_start+time.time())
    return model


def train_Baseline(train,test,lambda_item,lambda_user):
    mean=train.sum()/train.nnz
    bias_item=np.random.rand(train.shape[0])
    bias_user=np.random.rand(train.shape[1])
    model=Baseline_Model(bias_item,bias_user,mean)
    model.set_nz(train)
    max_epochs=7
    errors=[]
    for it in range(max_epochs):
        model.update_bias_item(train,lambda_item)
        model.update_bias_user(train,lambda_user)
        errors.append(model.compute_error(train))
        print("Training RMSE: ", errors[-1])

    if test is not None:
        print("Test RMSE: ",model.compute_error(test))
    return model


def train_SVD(train,test,lambda_item=40,lambda_user=50):
    """
        Returns SVD model trained by ALS
        Returns:
            model: SVD_LS_Model
    """
    num_features=20
    max_epochs=10   #max iteration count
    user_features,item_features=init_MF(train,num_features)
    model=SVD_LS_Model(item_features,user_features) #setup model
    model.set_nz(train)                             #Store nonzero indexes, don't forget to do this for any LS method
    for it in range(max_epochs):
        model.update_item_features(train,lambda_item) #update item_features
        model.update_user_features(train,lambda_user) #update_user_features
        print("Iteration ",it,", TRAINING RMSE: ",model.compute_error(train)) #print current training RMSE
    if test is not None:
        print("TEST RMSE: ",model.compute_error(test)) #print test RMSE
    return model


def train_SVD_using_SGD(train, test,lambda_user = 0.1,lambda_item = 0.7):
    """
    Args:
        train: training matrix
        test: test matrix
        lambda_user:
        lambda_item: regularization parameters
    Returns:
        model: SVD_Model
    """
    gamma = 0.05    #learning rate
    num_features = 20   # f

    num_epochs = 20     # number of full passes through the train set

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2
        for d, n in nz_train:
            #calculate current prediction
            currentM=item_features[d,:].dot(user_features[n,:])
            #calculate gradients
            item_grad=-(train[d,n]-currentM)*(user_features[n,:])+lambda_item*item_features[d,:]
            user_grad=-(train[d,n]-currentM)*(item_features[d,:])+lambda_user*user_features[n,:]
            #apply gradient descent
            item_features[d,:]-=gamma*item_grad
            user_features[n,:]-=gamma*user_grad
        #update model
        model=SVD_Model(item_features,user_features)
        #print training error for this iteration
        rmse=model.compute_error(train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
    #print test error
    if test is not None:
        rmse = model.compute_error(test)
        print("RMSE on test data: {}.".format(rmse))
    return model


def train_SVDpp(train,test):
    """
    Args:
        train: training matrix
        test:  test matrix
    Returns:
        model: SVDpp_Model
    """
    num_features=20
    gamma = 0.005        #learning rate
    lambda_user = 0.03
    lambda_item = 0.04
    lambda_bias= 0.05
    lambda_imp= 0.0
    num_epochs = 15     # number of iterations
    np.random.seed(988)
    # setup model
    user_features, item_features = init_MF(train, num_features)
    mean=train.sum()/train.nnz
    user_features=np.ones([train.shape[1],num_features])*0.1
    item_features=np.ones([train.shape[0],num_features])*0.1
    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    bias_item=np.zeros(train.shape[0])
    bias_user=np.zeros(train.shape[1])
    imp_factors=np.zeros([train.shape[0],num_features])
    model=SVD_Bias_Imp_Model(item_features,user_features,bias_item,bias_user,mean,imp_factors)
    model.set_nz(train)
    np_train=train.toarray()
    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        # decrease step size
        gamma *=0.95
        start_time=time.time()
        for n in range(np_train.shape[1]): #for each user
            items_n=model.nz_user[n][1]    #get list of items with nonzero ratings for that user
            if(len(items_n)<=0):           #skip user if there is no ratings on that user
                continue
            coeff=(len(items_n))**(-0.5)   #Normalizing coefficient sqrt(|\Omega_u|)
            sumVal=np.sum(imp_factors[items_n,:],axis=0)    #Sum(y_j) in given \Omega_u
            resVal=np.zeros([item_features.shape[1]])       #initialize resiudal gradient factor to 0
            for d in items_n:
                currentM=item_features[d,:].dot(user_features[n,:]+coeff*sumVal) #The term w_i(z_u+sum(y_j))
                err=(np_train[d,n]-currentM-bias_item[d]-bias_user[n]-mean)      #Error of predicition with current values
                bias_item[d]+=gamma*(err-lambda_bias*bias_item[d])               #update bias terms
                bias_user[n]+=gamma*(err-lambda_bias*bias_user[n])
                item_grad=-err*(user_features[n,:]+coeff*sumVal)+lambda_item*item_features[d,:] #update item and user features
                user_grad=-err*(item_features[d,:])+lambda_user*user_features[n,:]
                item_features[d,:]-=gamma*item_grad
                user_features[n,:]-=gamma*user_grad
                resVal+=err*item_features[d]                                      #add e_iu*w_i
            resVal/=len(items_n)   #Take the average for all the factors encountered for this user
            imp_factors[items_n]+=gamma*(coeff*resVal-lambda_imp*imp_factors[items_n]) #update y_j using the residual gradient term
        model=SVD_Bias_Imp_Model(item_features,user_features,bias_item,bias_user,mean,imp_factors)#update model
        model.set_nz(train)
        rmse=model.compute_error(train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        if(test is not None):
            rmse = model.compute_error(test)
            print("RMSE on test data: {}.".format(rmse))
        print("This iter in time:",time.time()-start_time)
    if (test is not None):
        rmse = model.compute_error(test)
        print("RMSE on test data: {}.".format(rmse))
    return model

def CCD_train(train,test,preModel=None, num_features=10, lambda_user=0.01, lambda_item=0.7, num_epochs=5):
    """
    Args:
        train: training matrix
        test: testing matrix
        preModel: a model to inherit from another iteration, if you are not using this for CCD++ it should be None
        num_features: f
        lambda_user:
        lambda_item: regularization parameters
        num_epochs: number of iterations
    Returns:
        model: CCD_Model
    """
    stop_criterion = 1e-4
    user_features,item_features=init_MF(train,num_features) #Initialize matrices
    residual = train - item_features @ user_features.T #calculate the resiudal matrix
    model=CCD_Model(item_features,user_features,residual) #setup model
    if(preModel is None):
        model.set_nz(train)
    else:
        model.nz_user=preModel.nz_user  #use nonzero indexes of previous model if there is any
        model.nz_item=preModel.nz_item
    errors=[]
    for it in range(num_epochs):
        model.update_user_feature_CCD(lambda_user)# update features
        model.update_item_feature_CCD(lambda_item)
        rmse = model.compute_error(train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        errors.append(rmse)
        if (len(errors) > 2) and (abs(errors[-1] - errors[-2]) < stop_criterion): #stop if the change in training RMSE is lower than stop_criterion
            break
    if(test is not None):           #print testing error if there is any test matrix
        print("Test RMSE: ",model.compute_error(test))
    return model
