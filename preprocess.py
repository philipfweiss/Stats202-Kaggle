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

def load_data_d(filenames, is_e=False):
    data = []
    all_labels = []
    for filename in filenames:
        with open(filename) as cv:
            vals = [row.split(',') for row in cv]
            nrows = len(vals)

            txgroup = {data[6] for idx, data in enumerate(vals) if idx != 0}
            tx_group_mapping = {k: idx for idx, k in enumerate(txgroup)}


## For each (patient, day) pair, keep track of:
##  - 10 last scores (P, N, G totals)
##  - Num previous observations
##  - TxGroup

NUM_PREDICTORS = 27
TEST_DAY = 126
countries = set(['"USA"', '"UK"', '"Russia"', '"India"', '"Spain"', '"Country"', '"Ukraine"', '"Czech Republic"', '"Romania"'] + ['"Russia"', '"Mexico"', '"Sweden"', '"Ukraine"', '"Korea"', '"Czech Republic"', '"Poland"', '"Bulgaria"', '"Greece"', '"Country"', '"Argentina"', '"Canada"', '"Japan"', '"Belgium"', '"Taiwan"', '"USA"', '"Spain"', '"China"', '"Australia"', '"Romania"', '"ERROR"', '"Brazil"', '"Slovakia"', '"Hungary"', '"Germany"', '"France"', '"Portugal"', '"Austria"'])
countryMap = {v: idx for idx, v in enumerate(countries)}

class Observation:
    def __init__(self, pid, observation_no, day, p, n, g):
        self.pid = pid
        self.observation_no = observation_no
        self.day = day
        self.p, self.n, self.g = p, n, g

class CPatient:
    def __init__(self, id, tx, ctr):
        self.ctr = countryMap[ctr]
        self.id = id
        self.tx = 0 if tx == "Control" else 1
        self.observations = []
        self.nullobs = Observation(self.id, 0, 0, 0, 0, 0)

    def add_observation(self, obs):
        self.observations.append(obs)

    def res(self, cur, first, second, third, fourth, fifth, day):
        fm = np.array([
            self.tx,
            day,
            first.observation_no, first.day, first.p, first.n, first.g,
            second.observation_no, second.day, second.p, second.n, second.g,
            third.observation_no, third.day, third.p, third.n, third.g,
            fourth.observation_no, fourth.day, fourth.p, fourth.n, fourth.g,
            fifth.observation_no, fifth.day, fifth.p, fifth.n, fifth.g,
        ])
        label = cur.p + cur.n + cur.g
        return fm, label, cur.pid

    def getFeatureMaps(self, is_e=False):
        ## On the test set, we just grab the last 4 observations.
        if is_e:
            num_ex = len(self.observations)
            first = self.observations[num_ex - 1] if num_ex > 0 else self.nullobs
            second = self.observations[num_ex - 1] if num_ex > 1 else self.nullobs
            third = self.observations[num_ex - 1] if num_ex > 2 else self.nullobs
            fourth = self.observations[num_ex - 1] if num_ex > 3 else self.nullobs
            fifth = self.observations[num_ex - 1] if num_ex > 4 else self.nullobs

            yield self.res(self.nullobs, first, second, third, fourth, fifth, TEST_DAY)
        else:
            for i, obs in enumerate(self.observations[1:]):
                cur = self.observations[i+1]
                first = self.observations[i]
                second = self.nullobs if (i < 2) else self.observations[i-1]
                third = self.nullobs if (i < 3) else self.observations[i-2]
                fourth = self.nullobs if (i < 4) else self.observations[i-3]
                fifth = self.nullobs if (i < 5) else self.observations[i-4]

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
            countries = {data[1] for data in vals}
            countryMap = {country: idx for idx, country in enumerate(countries)}
            patients = {(data[2], data[6], data[1]) for idx, data in enumerate(vals) if idx != 0}
            patientMap = {id: CPatient(id, tx, ctr) for idx, (id, tx, ctr) in enumerate(patients)}
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

            if is_e:
                num_observations = len(e_pids)
            else:
                num_observations = sum([patientCounter[p] for p in patients])


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
def train_val_test(dataset, train=.8, val=.15, test=.05):
    data, labels = dataset
    n, d = data.shape
    num_train, num_val, num_test = int(n * train), int(n * val), int(n * test)

    train = (data[0:num_train, :], labels[0:num_train])
    val = (data[num_train:num_train+num_val, :], labels[num_train:num_train+num_val])
    test = (data[num_train+num_val:, :], labels[num_train+num_val:])
    return train, val, test
