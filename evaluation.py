
import numpy as np
from sklearn.metrics import log_loss
def test_accuracy_binary_classification(model, dataset):
    with open('status.csv') as cv:
        vals = [row.split(',') for row in cv]
        obs_ids = [int(val[0]) for idx, val in enumerate(vals) if idx != 0]



    data, labels, ids = dataset
    n, d = data.shape
    prediction = model.predict_proba(data)
    labelmtx = np.zeros((n, 2))
    for idx, l in enumerate(labels):
        labelmtx[idx, int(l)] = 1

    resultMap = {}
    for idx, pred in enumerate(prediction):
        resultMap[ids[idx][0]] = prediction[idx, 1]

    for obs in obs_ids:
        print(resultMap[obs])

    return log_loss(labelmtx, prediction)

def test_accuracy_regression(model, dataset):
    data, labels, _ = dataset
    prediction = model.predict(data)
    for idx, p in enumerate(prediction):
        print(p)
        # if ((prediction[idx]- labels[idx])**2 > 100):
            # print(f"{p} - {labels[idx]} - {(p - labels[idx])**2}")
            # print(data[idx])
    RSS = np.average((prediction - labels)**2)
    # print(RSS)
    # for i in range(len(prediction)):
    #     print(f"pred: {prediction[i]}, labels: {labels[i]}")
    return RSS
