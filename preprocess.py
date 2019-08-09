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

## For each (patient, day) pair, keep track of:
##  - 10 last scores (P, N, G totals)
##  - Num previous observations
##  - TxGroup

NUM_PREDICTORS = 21
TEST_DAY = 120
countries = {
    '"UK"': 1,
    '"USA"': 2,
    '"Russia"': 3,
}

studies = {'"A"', '"B"', '"C"', '"D"', '"E"'}

class Observation:
    def __init__(self, pid, observation_no, day, p, n, g):
        self.pid = pid
        self.observation_no = observation_no
        self.day = day
        self.p, self.n, self.g = p, n, g

class CPatient:
    def __init__(self, id, tx, ctr, stdy):
        self.ctr = countries[ctr] if ctr in countries else 0
        self.study = stdy
        self.id = id
        self.tx = 0 if tx == '"Control"' else 1
        self.observations = []
        self.nullobs = Observation(self.id, 0, 0, 0, 0, 0)

    def add_observation(self, obs):
        self.observations.append(obs)

    def res(self, cur, first, second, third, fourth, fifth, day):
        ctr = [0, 0, 0, 0]
        ctr[self.ctr] = 1
        # print(*ctr)
        fm = np.array([
            self.tx,
            day,
            *[1 if self.study == study else 0 for study in studies],
            *ctr,
            fifth.day, fourth.day, third.day, second.day, first.day,
            first.p + first.n + first.g,
            second.p + second.n + second.g,
            third.p + third.n + third.g,
            fourth.p + fourth.n + fourth.g,
            fifth.p + fifth.n + fifth.g,

        ])

        label = cur.p + cur.n + cur.g
        return fm, label, cur.pid

    def getFeatureMaps(self, is_e=False):
        ## On the test set, we just grab the last 4 observations.
        if is_e:
            num_ex = len(self.observations)
            first = self.observations[num_ex - 1] if num_ex > 0 else self.nullobs
            second = self.observations[num_ex - 2] if num_ex > 1 else self.nullobs
            third = self.observations[num_ex - 3] if num_ex > 2 else self.nullobs
            fourth = self.observations[num_ex - 4] if num_ex > 3 else self.nullobs
            fifth = self.observations[num_ex - 5] if num_ex > 4 else self.nullobs

            yield self.res(self.nullobs, first, second, third, fourth, fifth, TEST_DAY)
        else:
            for i, obs in enumerate(self.observations[2:]):
                cur = self.observations[i+2]
                first = self.observations[i+1]
                second = self.observations[i]
                third = self.nullobs if (i < 2) else self.observations[i-1]
                fourth = self.nullobs if (i < 3) else self.observations[i-2]
                fifth = self.nullobs if (i < 4) else self.observations[i-3]

                yield self.res(cur, first, second, third, fourth, fifth, cur.day)



def load_data_c(filenames, e_pids=None):
    data = []
    all_labels = []
    pidmap = collections.defaultdict(list)

    for filename in filenames:
        with open(filename) as cv:

            ## Add some counting data structures.
            vals = [row.split(',') for row in cv]
            nrows = len(vals)
            patients = {(data[2], data[6], data[1], data[0]) for idx, data in enumerate(vals) if idx != 0}
            patientMap = {id: CPatient(id, tx, ctr, stdy) for idx, (id, tx, ctr, stdy) in enumerate(patients)}
            patientCounter = collections.defaultdict(int)

            ## Collect all of the observations.
            for idx, row in enumerate(vals):
                if idx == 0: continue
                p, n, g = computePNG(row)
                day, patient_no = int(row[7]), row[2]
                patientCounter[patient_no] += 1
                obs = Observation(patient_no, patientCounter[patient_no], day, p, n, g)
                patientMap[patient_no].add_observation(obs)

            ## Add data points and labels,
            is_e = (e_pids is not None)
            patients = [k for k, v in patientCounter.items() if v > 1]
            patients = (e_pids) if is_e else (patients)


            num_observations = 0
            for patient in patients:
                cpatient = patientMap[patient]
                for featuremap, label, pid in patientMap[patient].getFeatureMaps(is_e=is_e):
                    num_observations += 1

            mtx, labels = np.zeros((num_observations, NUM_PREDICTORS)), np.zeros((num_observations, 1))
            counter = 0

            for patient in patients:
                cpatient = patientMap[patient]
                for featuremap, label, pid in patientMap[patient].getFeatureMaps(is_e=is_e):
                    mtx[counter, :] = featuremap
                    labels[counter, :] = label
                    pidmap[pid].append(counter)
                    counter += 1
            data.append(mtx)
            all_labels.append(labels)

    combined_data, combined_labels = np.vstack(tuple(data)), np.vstack(tuple(all_labels))
    combined_data = sparse.csr_matrix(combined_data)
    print(combined_data.shape, is_e)
    return combined_data, combined_labels[:, 0], pidmap

def computePNG(row):
    p, n, g = 0, 0, 0
    for i in range(8, 15): p += float(row[i])
    for i in range(15, 22): n += float(row[i])
    for i in range(22, 38): g += float(row[i])
    return p, n, g


"""
input: (nxd) matrix of training data, (n,1) matrix of labels
returns: training, validation, testing set
"""
def train_val_test(dataset, train=.9, val=.1, test=.0):
    data, labels, ids = dataset
    n, d = data.shape
    is_numpy = type(ids).__module__ == np.__name__
    if not is_numpy:
        ids = np.zeros((n,))
    p = np.random.permutation(n)
    data, labels, ids = data[p], labels[p], ids[p]

    num_train, num_val, num_test = int(n * train), int(n * val), int(n * test)

    train = (data[0:num_train, :], labels[0:num_train], ids[0:num_train],)
    val = (data[num_train:num_train+num_val, :], labels[num_train:num_train+num_val], ids[num_train:num_train+num_val])
    test = (data[num_train+num_val:, :], labels[num_train+num_val:], ids[num_train+num_val:])
    return train, val, test
