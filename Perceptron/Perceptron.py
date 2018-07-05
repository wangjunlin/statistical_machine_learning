from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random


# 参考 https://blog.csdn.net/wds2006sdo/article/details/51923546

def Train(train_set, train_label):
    train_size = len(train_label)

    w = np.zeros((2, 1))  # w(x1,x2)
    b = 0

    study_count = 0
    nochange_count = 0
    nochange_upper_limit = 10000

    while True:
        nochange_count += 1
        if nochange_count > nochange_upper_limit:
            print('nochange_count > nochange_upper_limit')
            break

        # 随机选取数据
        index = random.randint(0, train_size - 1)
        train_data = train_set[index]
        # print('train_data', train_data)
        label = train_label[index]

        # 计算yi(w*xi+b)
        yi = label
        result = yi * (np.dot(train_data, w) + b)
        # print('result:', result)
        # 如果yi(w*xi+b)<=0 更新w,b值
        if result <= 0:
            train_data = np.reshape(train_set[index], (2, 1))
            print('train_data', train_data)
            w += train_data * yi * 1
            b += yi * 1

            print('w = {}, b = {}'.format(w, b))
            study_count += 1
            print('study count :', study_count)
            if study_count > 10000:
                print('study_count > 10000')
                break
            nochange_count = 0

    return w, b


def Predict(test_set, w, b):
    predict = []
    for X in test_set:
        result = np.dot(X, w) + b
        result = result > 0

        predict.append(result)
    return np.array(predict)


if __name__ == '__main__':
    print('Start!')
    load_data = load_iris().get('data')
    X = load_data[0:100, [0, 2]]
    # print(X)

    load_label = load_iris().get('target')[0:100]
    Y = [1 if x == 0 else -1 for x in load_label]

    train_feature, test_features, train_labels, test_label = train_test_split(X, Y, test_size=0.2, random_state=7)
    print('train_feature size : {}, test_feature size {}'.format(len(train_feature), len(test_features)))

    print('Start Training!')
    w, b = Train(train_feature, train_labels)

    print('Train Complete ! w = {}, b = {}'.format(w, b))

    test_predict = Predict(test_features, w, b)
    score = accuracy_score(test_label, test_predict)
    print('The accuracy score is：', score)
