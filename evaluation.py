
import numpy as np

def test_accuracy_binary_classification(model, dataset):
    pass

def test_accuracy_regression(model, dataset):
    data, labels, _ = dataset
    prediction = model.predict(data)
    RSS = np.average((prediction - labels)**2)
    for i in range(len(prediction)):
        print(f"pred: {prediction[i]}, labels: {labels[i]}")
    return RSS
