from sklearn import preprocessing
import numpy as np
import random
import csv
from scipy import sparse
import collections
import math
from sklearn.preprocessing import normalize

"""
Given a study, return the matrix of shape (nrows, ncols),
where each row is 'transform' applied to the corresponding row
in the dataset.
"""


def load_data(filenames, is_e=False):

    PHI_DIM = 33
    NUM_DAYS = 87

    data = []
    all_labels = []
    for filename in filenames:
        with open(filename) as cv:
            vals = [row.split(',') for row in cv]
            nrows = len(vals)

            txgroup = {data[6] for idx, data in enumerate(vals) if idx != 0}
            tx_group_mapping = {k: idx for idx, k in enumerate(txgroup)}

            ## Remove a patient if they never have have entry after 119 days.
            patient_max = collections.defaultdict(int)
            for idx, row in enumerate(vals):
                if idx == 0: continue
                day = int(row[7])
                patient_no = row[2]
                patient_max[patient_no] = max(patient_max[patient_no], day)


            patients = {patient for patient, maxday in patient_max.items() if maxday >= NUM_DAYS}
            patient_map = {k: idx for idx, k in enumerate(patients)}

            labels = np.zeros((len(patients), 1))
            mtx = np.zeros((len(patients), (NUM_DAYS)+2))
            for idx, row in enumerate(vals):
                if idx == 0: continue

                day = int(row[7])

                if row[2] not in patients: continue
                patient_no = patient_map[row[2]]

                if (day >= NUM_DAYS):
                    if (labels[patient_no] == 0):
                        labels[patient_no] = float(row[38])
                else:
                    mtx[patient_no, tx_group_mapping[row[6]]] = 1
                    mtx[patient_no, 2+day] = float(row[38])
                    # for i in range(9,39):
                    #     mtx[patient_no, 2+(31*day) + (i-9)] = float(row[i])
            
            data.append(mtx)
            all_labels.append(labels)

    combined_data, combined_labels = np.vstack(tuple(data)), np.vstack(tuple(all_labels))
    combined_data = sparse.csr_matrix(combined_data)
    return combined_data, combined_labels[:, 0]

"""
input: (nxd) matrix of training data, (n,1) matrix of labels
returns: training, validation, testing set
"""
def train_val_test(dataset, train=.8, val=.15, test=.05):
    data, labels = dataset
    n, d = data.shape
    num_train, num_val, num_test = int(n * train), int(n * val), int(n * test)

    train = (data[0:num_train, :], labels[0:num_train])
    val = (data[num_train:num_train+num_val, :], labels[num_train:num_train+num_val])
    test = (data[num_train+num_val:, :], labels[num_train+num_val:])
    return train, val, test
