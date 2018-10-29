import knn

learningData = [(5.1, 3.4, 1.5, 0.2, 'Iris-setosa'),
                (6.6, 2.9, 4.6, 1.3, 'Iris-versicolor'),
                (7.9, 3.8, 6.4, 2.0, 'Iris-virginica')]
obj = knn.Knn(3, learningData)


def testScore():
    data = [(5.1, 3.4, 1.5, 0.2, 'Iris-setosa'),
            (6.6, 2.9, 4.6, 1.3, 'Iris-versicolor'),
            (7.9, 3.8, 6.4, 2.0, 'Iris-virginica')]
    predictedLabels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    score = obj.score(data, predictedLabels)
    assert score == 1
