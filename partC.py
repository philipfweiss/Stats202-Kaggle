import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import sparse
from xgboost import XGBRegressor

import csv
from preprocess import *
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
def train_model(dataset):
    data, labels = dataset
    n, d = data.shape
    return XGBRegressor(early_stopping_rounds=5, gamma=0.1, max_depth=100, eval_metric="rmse").fit(data, labels)


# def get_fake_data():
#     x = np.random.normal(0, 1, (20000, 1))
#     e = np.random.normal(0, .1, (20000, 1))
#     y = x + e
#     plt.scatter(x, y)
#     plt.show()
#     return x, y


def part_c():
    all_data = load_data(['Study_A.csv', 'Study_B.csv', 'Study_C.csv', 'Study_D.csv']) ## TODO: Change back to actual data loading.
    test = load_data(['Study_E.csv'], is_e=True)
    print(all_data[0].shape)
    print(test[0].shape)
    # train, val, test = train_val_test(all_data)
    model = train_model(all_data)
    # test_accuracy_regression(model, all_data)
    test_accuracy_regression(model, test)



def part_d():
    pass


def __main__():
    part_c()
    # part_d()


__main__()
