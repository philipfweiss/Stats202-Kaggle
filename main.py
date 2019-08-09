import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import sparse
from xgboost import XGBRegressor
from sklearn import linear_model
import csv
from preprocess import *
from preprocess_d import *

from evaluation import *

"""
Questions:
    - Are we using data from all studies in our prediction, or just in one study?
    - Which feature encoding should we use? (See preprocess.py)
    - Which classification algorithm should we use?
"""


"""
Input: (n x d) matrix representing data in the training set,
which has already been pre-processed and is ready to train on.
Output: Trained model.
"""
def train_model(dataset, gamma, depth):
    data, labels, _ = dataset
    n, d = data.shape
    return XGBRegressor(gamma=gamma, max_depth=depth, objective='reg:squarederror').fit(data, labels)

    # return XGBRegressor(gamma=47.71, max_depth=5, objective='reg:squarederror').fit(data, labels)

def train_model_d(dataset):
    data, labels, _ = dataset
    n, d = data.shape
    return XGBRegressor(gamma=gamma, max_depth=depth, objective='binary:logistic').fit(data, labels)


def part_c():

    e_pids = None
    with open('sample_submission_PANSS.csv') as cv:
        vals = [row.split(',') for row in cv]
        e_pids = [val[0] for idx, val in enumerate(vals) if idx != 0]


    all_data = load_data_c(['Study_A.csv', 'Study_B.csv', 'Study_C.csv', 'Study_D.csv', 'Study_E.csv']) ## TODO: Change back to actual data loading.
    train, val, _ = train_val_test(all_data)
    test = load_data_c(['Study_E.csv'], e_pids=e_pids)
    model = train_model(all_data, 42.71, 5)
    acc = test_accuracy_regression(model, test)
    print(acc)
    #
    # loss = []
    # for gamma in np.linspace(32, 62, 15):
    #     model = train_model(train, gamma, 5)
    #     acc = test_accuracy_regression(model, val)
    #     print(acc, gamma, 5)
    #     loss.append((acc, gamma, 5))
    # print(sorted(loss))



def part_d():
    all_data = load_data_d(['Study_A.csv', 'Study_B.csv', 'Study_C.csv', 'Study_D.csv']) ## TODO: Change back to actual data loading.
    train, val, _ = train_val_test(all_data)
    # test = load_data_c(['Study_E.csv'], e_pids=e_pids)
    model = train_model(all_data)
    # acc = test_accuracy_regression(model, test)
    # print(acc)


def __main__():
    # part_c()
    part_d()


__main__()
