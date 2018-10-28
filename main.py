import pandas as pd
import numpy as np
import knn

learningDataArray = np.array(pd.read_csv("iris.datalearning.csv", header=None))
testDataArray = np.array(pd.read_csv("iris.datatest.csv", header=None))
k = 3
kn = knn.Knn(k, learningDataArray)
predictedLabels = []
predictedLabels=kn.predict(testDataArray)
print(predictedLabels)

