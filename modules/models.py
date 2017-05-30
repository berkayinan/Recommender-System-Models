import numpy as np
from modules.helpers import build_index_groups

"""
#### About models object
Each method returns an object which is inherits the same class `SVD_Model`.
A model object inherits the following methods. (Implementation of methods for each model is different)
model.predict()-> returns prediction matrix
model.compute_error(data)->returns RMSE on the data
Models may have different methods based on their extensions.
"""

class Baseline_Model():
    """Class for baseline model"""
    def __init__(self,bias_item,bias_user,mean):
        self.bias_item=bias_item
        self.bias_user=bias_user
        self.mean=mean

    def predict(self):
        """Returns prediction matrix"""
        bias_user_mat = np.tile(self.bias_user, [self.bias_item.shape[0],1])
        bias_item_mat = np.tile(self.bias_item, [self.bias_user.shape[0],1]).T
        prediction = bias_item_mat + bias_user_mat + self.mean
        # Trim the scores into 1-5 range
        prediction[np.where(prediction > 5)] = 5
        prediction[np.where(prediction < 1)] = 1
        return prediction

    def set_nz(self,train):
        """Set nonzero indexes for given ratings matrix @train"""
        _, self.nz_item, self.nz_user = build_index_groups(train)

    def update_bias_item(self,train,lambda_item):
        """update self.bias_item vector based on @train data and regularization parameter @lambda_item"""
        for item_ind,users in self.nz_item:
            np_train=np.squeeze(train[item_ind,users].toarray())
            self.bias_item[item_ind]=np.sum(np_train-self.bias_user[users]-self.mean)/(len(users)+lambda_item)

    def update_bias_user(self,train,lambda_user):
        """update self.bias_user vector based on @train data and regularization parameter @lambda_user"""
        for user_ind, items in self.nz_user:
            np_train = np.squeeze(train[items,user_ind].toarray())
            self.bias_user[user_ind] = np.sum(np_train - self.bias_item[items] - self.mean)/(len(items)+lambda_user)

    def compute_error(self, data):
        """Returns RMSE on given data set"""
        prediction = self.predict()
        n_rows, n_cols = data.nonzero()
        rmse = np.mean((data[n_rows, n_cols].toarray() - prediction[n_rows, n_cols]) ** 2) ** 0.5
        return rmse


class SVD_Model():
    """Base model for SVD (matrix factorization) models"""
    def __init__(self,item_features,user_features):
        self.item_features=item_features
        self.user_features=user_features

    def set_nz(self,train):
        """Store nonzero indexes of @train matrix"""
        _,self.nz_item,self.nz_user=build_index_groups(train)

    def predict(self):
        """Returns prediction matrix"""
        prediction=self.item_features.dot(self.user_features.T)
        #Trim the scores to 1-5 range
        prediction[np.where(prediction>5)]=5
        prediction[np.where(prediction<1)]=1
        return prediction

    def compute_error(self,data):
        """Returns RMSE on given data set"""
        prediction=self.predict()
        n_rows,n_cols=data.nonzero()
        rmse=np.mean((data[n_rows,n_cols].toarray()-prediction[n_rows,n_cols])**2)**0.5
        return rmse
    

class SVD_Bias_Model(SVD_Model):
    """Model that expands upon SVD_Model"""
    def __init__(self,item_features,user_features,bias_item,bias_user,mean):
        super(SVD_Bias_Model,self).__init__(item_features,user_features)
        self.bias_item=bias_item
        self.bias_user=bias_user
        self.mean=mean
        
    def predict(self):
        """Returns prediction matrix"""
        bias_user_mat=np.tile(self.bias_user,[self.item_features.shape[0],1])
        bias_item_mat=np.tile(self.bias_item,[self.user_features.shape[0],1]).T
        prediction=self.item_features.dot(self.user_features.T)+bias_item_mat+bias_user_mat+self.mean
        #Trim the scores to 1-5 range
        prediction[np.where(prediction>5)]=5
        prediction[np.where(prediction<1)]=1
        return prediction


class SVD_LS_Model(SVD_Model):
    """Model that solves SVD with ALS"""
    def update_item_features(self,train,lambda_item):
        """Updates self.item_features matrix based on @train data and regularization parameter @lambda_item"""
        if self.nz_item is None:
            print("NZ NOT SET!\n")
            return
        for item_ind,users in self.nz_item:
            np_train=train[item_ind,users].toarray().T #convert the indexed part of train into np.ndarray
            residual=np_train
            Z_nz=self.user_features[users,:] #get Z_\Omega as described in report
            first_term=Z_nz.T.dot(Z_nz)+lambda_item*np.eye(self.item_features.shape[1])
            second_term=Z_nz.T.dot(residual)
            self.item_features[item_ind,:]=np.squeeze(np.linalg.solve(first_term,second_term).T)# solve least squares
    def update_user_features(self,train,lambda_user):
        """Updates self.user_features matrix based on @train data and regularization parameter @lambda_user"""
        if self.nz_user is None:
            print("NZ NOT SET!\n")
            return
        for user_ind,items in self.nz_user:
            np_train=train[items,user_ind].toarray()
            residual=np_train
            W_nz=self.item_features[items,:]
            first_term=W_nz.T.dot(W_nz)+lambda_user*np.eye(self.item_features.shape[1])
            second_term=W_nz.T.dot(residual)
            result=np.linalg.solve(first_term,second_term)
            self.user_features[user_ind,:]=np.squeeze(result)

class SVD_Bias_LS_Model(SVD_Bias_Model):
    """Model that uses bias and mean terms in SVD models"""
    def update_item_features(self,train,lambda_item,lambda_bias):
        """Updates self.item_features and self.bias_item matrix based on @train data and regularization parameter @lambda_bias"""
        if self.nz_item is None:
            print("NZ NOT SET!\n")
            return
        for item_ind,users in self.nz_item:
            np_train=np.squeeze(train[item_ind,users].toarray())
            residual=np_train-self.bias_item[item_ind]-self.bias_user[users]-self.mean
            Z_nz=self.user_features[users,:]
            first_term=Z_nz.T.dot(Z_nz)+lambda_item*np.eye(self.item_features.shape[1])
            second_term=Z_nz.T.dot(residual)
            self.item_features[item_ind,:]=np.squeeze(np.linalg.solve(first_term,second_term).T)
            self.bias_item[item_ind]=np.sum(np_train-self.bias_user[users]-self.item_features[item_ind,:].dot(Z_nz.T)-self.mean)/(len(users)+lambda_bias)

    def update_user_features(self,train,lambda_user,lambda_bias):
        """Updates self.user_features and self.bias_user matrix based on @train data and regularization parameter @lambda_bias"""
        if self.nz_user is None:
            print("NZ NOT SET!\n")
            return
        for user_ind,items in self.nz_user:
            np_train=np.squeeze(train[items,user_ind].toarray())
            residual=np_train-self.bias_item[items]-self.bias_user[user_ind]-self.mean
            W_nz=self.item_features[items,:]
            first_term=W_nz.T.dot(W_nz)+lambda_user*np.eye(self.item_features.shape[1])
            second_term=W_nz.T.dot(residual)
            result=np.linalg.solve(first_term,second_term)
            self.user_features[user_ind,:]=np.squeeze(result)
            self.bias_user[user_ind]=np.sum(np_train-self.bias_item[items]-W_nz.dot(result)-self.mean)/(len(items)+lambda_bias)


class SVD_Bias_Imp_Model(SVD_Bias_Model):
    """Model for SVD with bias and implicit feedback (SVD++)"""
    def __init__(self,item_features,user_features,bias_item,bias_user,mean,imp_factors):
        super(SVD_Bias_Imp_Model,self).__init__(item_features,user_features,bias_item,bias_user,mean)
        self.imp_factors=imp_factors
    def predict(self):
        bias_user_mat=np.tile(self.bias_user,[self.item_features.shape[0],1])
        bias_item_mat=np.tile(self.bias_item,[self.user_features.shape[0],1]).T
        new_user_features=self.user_features.copy()
        for user_ind,items in self.nz_user:
            coeff=len(items)**(-0.5)
            new_user_features[user_ind]+=coeff*np.sum(self.imp_factors[items],axis=0)
        prediction=self.item_features.dot(new_user_features.T)+bias_item_mat+bias_user_mat+self.mean        
        prediction[np.where(prediction>5)]=5
        prediction[np.where(prediction<1)]=1
        return prediction
    
    
class CCD_Model(SVD_Model):
    """Model for CCD"""
    def __init__(self,item_features,user_features,residual):
        super(CCD_Model,self).__init__(item_features,user_features)
        self.residual=residual
        return
    def update_user_feature_CCD(self, lambda_user):
        """update user feature matrix."""
        if(self.nz_user is None):
            raise Exception
        for n,nz_idx in self.nz_user:
            for k in range(self.user_features.shape[1]):
                nom = (self.residual[nz_idx, n] + self.user_features[n,k] * self.item_features[nz_idx, k]).T @ self.item_features[nz_idx, k]
                denom = lambda_user + self.item_features[nz_idx, k].T @ self.item_features[nz_idx, k]
                s = (nom / denom)[0,0] # (12) in paper
                #update R and H, (9) and (10) in paper
                self.residual[nz_idx, n] = np.squeeze(self.residual[nz_idx, n] - ((s - self.user_features[n,k]) * np.c_[self.item_features[nz_idx, k]]))
                self.user_features[n,k] = s

    def update_item_feature_CCD(self, lambda_item):
        """update item feature matrix."""
        if(self.nz_item is None):
            raise Exception
        for d,nz_idx in self.nz_item:
            for k in range(self.item_features.shape[1]):
                nom = (self.residual[d, nz_idx] + self.item_features[d, k] * self.user_features[nz_idx,k].T @ self.user_features[nz_idx, k])
                denom = lambda_item + self.user_features[nz_idx,k].T @ self.user_features[ nz_idx,k]
                z = (nom / denom)[0,0] # (6) in paper            
                #update R and W, (7) and (8) in paper
                self.residual[d, nz_idx] = np.squeeze(self.residual[d, nz_idx] - (z - self.item_features[d, k]) * np.r_[self.user_features[nz_idx,k]])
                self.item_features[d, k] = z


class CCDPP_Model(CCD_Model):
    """Model for CCD++"""
    def set_method(self,method):
        self.method=method
    def update_matrices(self,train):
        num_features=self.item_features.shape[1]
        for k in range(num_features):    
            print(" k={}/{}...".format(k + 1, num_features + 1))
            residual_hat = self.residual + (np.c_[self.item_features[:, k]] @ np.c_[self.user_features[ :,k]].T)
            newModel = self.method(residual_hat, None, preModel=self,num_features=1)
            residual = residual_hat - newModel.item_features @ newModel.user_features.T
            self.user_features[:, k] = newModel.user_features
            self.item_features[:, k] = newModel.item_features
