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

NUM_PREDICTORS = 20
countries = {
    '"USA"':1,
    '"UK"':2,
    '"Russia"':3,
}

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
        self.ctr = countries[ctr] if ctr in countries else 0
        self.study = stdy
        self.id = id
        self.tx = 0 if tx == '"Control"' else 1
        self.observations = []
        self.nullobs = Observation(self.id, 0, 0, 0, 0, 0, 0)

    def add_observation(self, obs):
        self.observations.append(obs)

    def res(self, cur, first, second, third, fourth, fifth):
        num_diff =  len(self.observations) - 1
        num_days = abs(self.observations[-1].day - self.observations[0].day)
        if num_diff > 0:
            avg_p_diff = sum(
                [self.observations[idx+1].p - self.observations[idx].p for idx, _ in enumerate(self.observations) if idx < num_diff]
            ) / num_diff
            avg_n_diff = sum(
                [self.observations[idx+1].n - self.observations[idx].n for idx, _ in enumerate(self.observations) if idx < num_diff]
            ) / num_diff
            avg_g_diff = sum(
                [self.observations[idx+1].g - self.observations[idx].g for idx, _ in enumerate(self.observations) if idx < num_diff]
            ) / num_diff
        else:
            avg_p_diff, avg_n_diff, avg_g_diff = 0, 0, 0

        p_change_per_day = abs(self.observations[-1].p - self.observations[0].p) / (num_days+1)
        n_change_per_day = abs(self.observations[-1].n - self.observations[0].n) / (num_days+1)
        g_change_per_day = abs(self.observations[-1].g - self.observations[0].g) / (num_days+1)

        # print(*[1 if self.ctr == idx else 0 for idx in range(len(countryMap.values()))])

        ctrys = [0, 0, 0, 0]
        ctrys[self.ctr] = 1
        fm = np.array([
            self.tx,
            cur.day,
            avg_p_diff, avg_n_diff, avg_g_diff,
            # p_change_per_day, n_change_per_day, g_change_per_day,
            *ctrys,
            fifth.day, fourth.day, third.day, second.day, first.day,
            cur.p + cur.n +cur.g,
            first.p + first.n + first.g,
            second.p + second.n + second.g,
            third.p + third.n + third.g,
            fourth.p + fourth.n + fourth.g,
            fifth.p + fifth.n + fifth.g,
        ])

        label = cur.label
        return fm, label, cur.observation_no

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
    "AssigntoCS": 1
}
def load_data_d(filenames, e_pids=None):
    data = []
    all_labels = []
    all_pids = []
    pidmap = collections.defaultdict(list)

    for filename in filenames:
        with open(filename) as cv:

            ## Add some counting data structures.
            vals = [row.split(',') for row in cv]
            nrows = len(vals)
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
                obs = Observation(patient_no, label, int(row[5]), day, p, n, g)
                patientMap[patient_no].add_observation(obs)

            ## Add data points and labels,
            is_e = (e_pids is not None)
            patients = [k for k, v in patientCounter.items() if v > 1]
            patients = (e_pids) if is_e else (patients)


            num_observations = 0
            for patient in patientMap.values():
                for featuremap, label, pid in patient.getFeatureMaps(is_e=is_e):
                    num_observations += 1

            mtx = np.zeros((num_observations, NUM_PREDICTORS))
            labels = np.zeros((num_observations, 2))
            pids = np.zeros((num_observations,1))

            counter = 0

            for patient in patientMap.values():
                for featuremap, label, pid in patient.getFeatureMaps(is_e=is_e):
                    mtx[counter, :] = featuremap
                    labels[counter, :] = label
                    pids[counter] = pid
                    counter += 1
            data.append(mtx)
            all_labels.append(labels)
            all_pids.append(pids)

    combined_data, combined_labels, combined_pids = np.vstack(tuple(data)), np.vstack(tuple(all_labels)),  np.vstack(tuple(all_pids))
    combined_data = sparse.csr_matrix(combined_data)
    # print(combined_data.shape, is_e)
    return combined_data, combined_labels[:, 0], combined_pids

def computePNG(row):
    p, n, g = 0, 0, 0
    for i in range(8, 15): p += float(row[i])
    for i in range(15, 22): n += float(row[i])
    for i in range(22, 38): g += float(row[i])
    return p, n, g
