
import numpy as np

def test_accuracy_binary_classification(model, dataset):
    pass

def test_accuracy_regression(model, dataset):
    data, labels = dataset
    prediction = model.predict(data)
    # for i in range(len(prediction)):
    #     print(prediction[i], labels[i])
    for i in range(len(prediction)):
        print(f"prediction: {prediction[i]}, truth: {labels[i]}")
    RSS = np.average((prediction - labels)**2)
    print(RSS)
