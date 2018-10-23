from sklearn.model_selection import train_test_split
import numpy as np

dataset = 'E:/python_workspace/CNN-Regression-Model/data/target_indel_20.csv'
train = 'E:/python_workspace/CNN-Regression-Model/data/train3.csv'
test = 'E:/python_workspace/CNN-Regression-Model/data/test3.csv'


with open(dataset, "r") as f:
    data = f.read().split('\n')
    data = np.array(data)
    x_train, x_test = train_test_split(data, test_size=0.1)

    with open(train, 'w') as train:
        for x in x_train:
            train.write(x + '\n')
        train.write('\r')
    train.close()

    with open(test, 'w') as test:
        for x in x_test:
            test.write(x + '\n')
        test.write('\r')
    test.close()
f.close()