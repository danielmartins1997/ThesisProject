import numpy as np
from sklearn import metrics

# Calculate the purity score of a clustering result
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    purity_score = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

    return purity_score

# Calculate the inverse purity score of a clustering result
def inverse_purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    inverse_purity_score = np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix) 

    return inverse_purity_score
