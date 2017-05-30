from modules.helpers import *
import csv
import pickle
def get_query(sampleSubmissionPath):
    """Get indexes that are asked for a given test data set"""
    sampleSub=load_data(sampleSubmissionPath)
    query_rows,query_cols=sampleSub.nonzero()
    return list(zip(query_rows,query_cols))

def create_submission(model,query,filename):
    """Use @model to predict the test @query and write them into given @filename in csv format"""
    prediction=model.predict()
    with open(filename,"w") as csvfile:
        writer=csv.writer(csvfile, lineterminator='\n')
        writer.writerow(["Id","Prediction"])
        for q_row,q_col in query:
            writer.writerow(["r"+str(q_row+1)+"_c"+str(q_col+1),prediction[q_row,q_col]])

