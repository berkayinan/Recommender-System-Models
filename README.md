Before running run.py, make sure that you have "sampleSubmssions.csv" in the input directory. That file includes the query indexes that is asked in the competition.
		run.py -train filename.csv
			This call starts the training process.
			It outputs 'submission.csv' for the query indexes.
			It also outputs a 'model.pkl' file for trained model. It can be loaded using -load option.
		run.py -load model.pkl
			This call tries to loads up the given model.pkl for an already trained model. It outputs 'submission.csv' for the query indexes.


Following files should be in the root:

sampleSubmission.csv
run.py: With train option it trains SVD_Bias_Model with train_SVD_Bias(train=ratings,test=None,lambda_user=39,lambda_item=50,lambda_bias=0,num_features=70,num_epochs=10,disable_mean=True)
	With load option it loads up a given model file.
	It outputs submission.csv.

cross_validation.py: Not run in the main algorithm. Only used for testing parameters. Runs cross validations on multiple threads.
		     Run by python cross_validation.py 
helpers.py : Contains methods for data loading
models.py : Contains model objects for each model. Each model has predict() and compute_error() methods. More in comments. 
submit.py : Contains model creating submission file
trainers.py : Contains training algorithms for models in models.py
utils.py : Contains utility methods

Optionally:
	Training file: datatrain.csv

We use numpy and scipy as external libraries. 