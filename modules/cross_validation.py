from modules.models import *
from modules.utils import *
from modules.helpers import *
from modules.trainers import *
import concurrent.futures
import time

class cross_validator():
    """Class for running cross validation in multiple threads"""
    def parallel_wrapper(self,k,data,method,args,k_indices):
        """Runs the k'th set for cross validation"""
        train,test=split_K_fold(data,k,k_indices)
        model=method(train,test,*args)
        return model.compute_error(test)

    def run(self,data,method,args_method,K_fold):
        """Runs the cross validation for given @method and @args_method for @K_fold"""
        k_indices=build_k_indices(data.nnz,K_fold,42)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            args_list = []
            for k in range(K_fold):
                args_list.append((k,data,method,args_method,k_indices))
            results=[]
            future_to_results = {executor.submit(self.parallel_wrapper, *args): args for args in args_list}
            for future in concurrent.futures.as_completed(future_to_results):
                results.append(future.result())
        return results

if __name__=='__main__':
    """Sample test of cross validation"""
    start_time=time.time()
    path_dataset = "data_train.csv"
    ratings = load_data(path_dataset)
    args_method=[20,10]
    l_time=time.time()
    print("Time elapsed during load data: ",l_time-start_time)
    results=cross_validator().run(ratings,train_ALS,args_method,4)
    print("Errors: ",results)
    print("Avg Error: ",np.mean(results))
    print("Elapsed Time: ",time.time()-start_time)
