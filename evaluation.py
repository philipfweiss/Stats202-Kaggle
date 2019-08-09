
import numpy as np
from sklearn.metrics import log_loss
def test_accuracy_binary_classification(model, dataset):
    data, labels, _ = dataset
    prediction = model.predict_proba(data)
    return log_loss(prediction, labels)

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
