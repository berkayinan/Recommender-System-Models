from modules.submit import *
from modules.trainers import *
import sys
import pickle
if __name__ == '__main__':
	"""
		Before running run.py, make sure that you have "sampleSubmssions.csv" in the input directory. That file includes the query indexes that is asked in the competition.
		run.py -train filename.csv
			This starts the training process.
			It outputs 'submission.csv' for the query indexes.
			It also outputs a 'model.pkl' file for trained model. It can be loaded using -load option.
		run.py -load model.pkl
			This tries to loads up the given model.pkl for an already trained model. It outputs 'submission.csv' for the query indexes.

	"""
	sampleSubmissionPath='input\sampleSubmission.csv' #Hard coded for submission
	if len(sys.argv) >= 2:
		if sys.argv[1] == '-load':
			print("Loading existing model")
			model=pickle.load(open(sys.argv[2],"rb"))	#load up the given model pkl file
		else:
			print("Training...")
			training_path=sys.argv[2]
			ratings=load_data(training_path)		#load data from .csv file
			model=train_SVD_Bias(train=ratings,test=None,lambda_user=39,lambda_item=50,lambda_bias=0,num_features=70,num_epochs=10,disable_mean=True)	#train model with no test data,use all data for training with no mean in calculations
			with open('model.pkl', 'wb') as output:	#create a new model
				pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
		print("Reading 'sampleSubmission.csv' file for indexes")
		query=get_query(sampleSubmissionPath)	#Get query indexes
		create_submission(model, query, 'submission.csv') #Create submission file
	else:
		print('Missing command line argument')

