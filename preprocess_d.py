from sklearn import preprocessing
import numpy as np
import random
import csv
from scipy import sparse
import collections
import math
from sklearn.preprocessing import normalize
import re

"""
Given a study, return the matrix of shape (nrows, ncols),
where each row is 'transform' applied to the corresponding row
in the dataset.
"""

## For each (patient, day) pair, keep track of:
##  - 10 last scores (P, N, G totals)
##  - Num previous observations
##  - TxGroup

NUM_PREDICTORS = 17
countries = set(['"USA"', '"UK"', '"Russia"', '"India"', '"Spain"', '"Country"', '"Ukraine"', '"Czech Republic"', '"Romania"'] + ['"Russia"', '"Mexico"', '"Sweden"', '"Ukraine"', '"Korea"', '"Czech Republic"', '"Poland"', '"Bulgaria"', '"Greece"', '"Country"', '"Argentina"', '"Canada"', '"Japan"', '"Belgium"', '"Taiwan"', '"USA"', '"Spain"', '"China"', '"Australia"', '"Romania"', '"ERROR"', '"Brazil"', '"Slovakia"', '"Hungary"', '"Germany"', '"France"', '"Portugal"', '"Austria"'])
countryMap = {v: idx for idx, v in enumerate(countries)}
studies = {'"A"', '"B"', '"C"', '"D"', '"E"'}

class Observation:
    def __init__(self, pid, label, observation_no, day, p, n, g):
        self.pid = pid
        self.label = label
        self.observation_no = observation_no
        self.day = day
        self.p, self.n, self.g = p, n, g

class DPatient:
    def __init__(self, id, tx, ctr, stdy):
        self.ctr = countryMap[ctr]
        self.study = stdy
        self.id = id
        self.tx = 0 if tx == '"Control"' else 1
        self.observations = []
        self.nullobs = Observation(self.id, 0, 0, 0, 0, 0, 0)

    def add_observation(self, obs):
        self.observations.append(obs)

    def res(self, cur, first, second, third, fourth, fifth):

        fm = np.array([
            self.tx,
            cur.day,
            *[1 if self.study == study else 0 for study in studies],
            *[1 if self.ctr == idx else 0 for idx in range(len(countries))],
            fifth.day, fourth.day, third.day, second.day, first.day,
            cur.p, cur.n, cur.g,
            first.p, first.n, first.g,
            second.p, second.n, second.g,
            third.p, third.n, third.g,
            fourth.p, fourth.n, fourth.g,
            fifth.p, fifth.n, fifth.g,
        ])

        label = cur.label
        return fm, label, cur.pid

    def getFeatureMaps(self, is_e=False):
        for i, obs in enumerate(self.observations):
            cur = self.observations[i]
            first = self.nullobs if (i < 1) else self.observations[i-1]
            second = self.nullobs if (i < 2) else self.observations[i-2]
            third = self.nullobs if (i < 3) else self.observations[i-3]
            fourth = self.nullobs if (i < 4) else self.observations[i-4]
            fifth = self.nullobs if (i < 5) else self.observations[i-5]
            yield self.res(cur, first, second, third, fourth, fifth)


options = {
    "Flagged": 1,
    "Passed": 0,
    "Assign to CS": 1
}
def load_data_d(filenames, e_pids=None):
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
            patients = {(data[2], data[6], data[1], data[0]) for idx, data in enumerate(vals) if idx != 0}
            patientMap = {id: DPatient(id, tx, ctr, stdy) for idx, (id, tx, ctr, stdy) in enumerate(patients)}
            patientCounter = collections.defaultdict(int)

            ## Collect all of the observations.
            for idx, row in enumerate(vals):
                if idx == 0: continue
                p, n, g = computePNG(row)
                day, patient_no = int(row[7]), row[2]
                patientCounter[patient_no] += 1
                label = options[re.sub(r'\W+', '', row[39])] if len(row) > 39 else 0
                obs = Observation(patient_no, label, patientCounter[patient_no], day, p, n, g)
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
    data, labels, _ = dataset
    n, d = data.shape
    p = np.random.permutation(n)
    data, labels = data[p], labels[p]

    num_train, num_val, num_test = int(n * train), int(n * val), int(n * test)

    train = (data[0:num_train, :], labels[0:num_train], 1)
    val = (data[num_train:num_train+num_val, :], labels[num_train:num_train+num_val], 1)
    test = (data[num_train+num_val:, :], labels[num_train+num_val:], 1)
    return train, val, test
