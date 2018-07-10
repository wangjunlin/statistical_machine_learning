"""
实现《统计学习方法》--李航 第八章 AdaBoost算法的例题
参考：https://blog.csdn.net/wds2006sdo/article/details/53195725#commentBox
"""
import math
import matplotlib.pyplot as plt
import numpy as np


class Sign(object):
    def __init__(self, features, labels, w):
        """
        该分类器就是一个二值分类器，同感知机的Sign(x)函数
        :param features: 特征 
        :param labels: 标签
        :param w: 权重
        """
        self.X = list(features)
        self.Y = labels
        self.N = len(labels)  # 训练数据大小
        self.w = w
        self.index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 阈值可选取范围
        self.save_index = []
        self.save_is_less = []

    # 寻找x小于阈值时label=1的最优阈值v
    def find_less(self):
        index = -1
        error_score = 10000

        # 对于阈值可选范围内的数逐一选取，取其中分类误差最小的阈值
        for i in self.index:
            score = 0
            for j in range(self.N):
                val = -1
                # 当数据中特征小于阈值，则标签置为1
                if self.X[j] < i:
                    val = 1
                # 分类结果和标签不一致（分类错误）时，score+1
                if val * self.Y[j] < 0:
                    score += self.w[j]
            # 每次得到的index 和 error_score和上一次的相比较，如果此次错误次数少，则最好的阈值为当前阈值
            if score < error_score:
                index = i
                error_score = score
        return index, error_score

    def find_more(self):
        index = -1
        error_score = 10000

        for i in self.index:
            score = 0
            for j in range(self.N):
                val = 1
                # 与上面的相同，只不过小于阈值的数label为-1
                if self.X[j] < i:
                    val = -1
                if val * self.Y[j] < 0:
                    score += self.w[j]
            if score < error_score:
                index = i
                error_score = score
        return index, error_score

    def train(self):
        # 训练分类器找出当前权重下最优阈值
        less_index, less_score = self.find_less()
        more_index, more_score = self.find_more()

        # 通过训练，得到最好的阈值
        if less_score < more_score:
            self.is_less = True  # 加入标志位，为后面的预测做判断
            self.index = less_index
            self.save_index.append(self.index)
            self.save_is_less.append(self.is_less)
            print('index:', self.index, 'less_score', less_score)
            return less_score
        else:
            self.is_less = False
            self.index = more_index
            self.save_index.append(self.index)
            self.save_is_less.append(self.is_less)
            print('index:', self.index, 'more_score:', more_score)
            return more_score

    def predict(self, feature):

        if self.is_less:
            if feature < self.index:
                return 1.0
            else:
                return -1.0
        else:
            if feature < self.index:
                return -1.0
            else:
                return 1.0


class AdaBoost:
    def __init__(self):
        pass

    def _parameters_(self, features, labels):
        self.X = features  # 训练集特征
        self.Y = labels  # 训练集标签
        self.n = 1  # 维数
        self.N = len(features)  # 训练集大小
        self.M = 3  # 分类器个数
        self.w = [1.0 / self.N] * self.N  # 初始化权重向量D1 = [w(1i)] i取1-10
        self.alpha = []  # 分类器Gi(x)的系数
        self.classifier = []  # 分类器

    def _w_(self, index, classifier, i):
        """
        计算 w[i+1] = w[i]exp(-alpha[i]*y[i]*classifier[i]):
        :return: w[i+1]为下一个分类器的权重
        """
        return self.w[i] * math.exp(-self.alpha[-1] * self.Y[i] * classifier.predict(self.X[i]))

    def _Z_(self, index, classifier):
        """
        计算规范化因子Zm
        :param index: 维度特征（这里用不着）
        :param classifier: 分类器
        :return: 
        """
        Z = 0
        for i in range(self.N):
            Z += self._w_(index, classifier, i)
        return Z

    def train(self, features, labels):
        self._parameters_(features, labels)
        # 生成分类器
        for epoch in range(self.M):
            best_classifier = (100000, None, None)  # 误差率，维度特征，分类器
            for i in range(self.n):
                if self.n > 1:
                    features = map(lambda x: x[i], self.X)  # 取出每个维度特征
                classifier = Sign(features, self.Y, self.w)
                error_score = classifier.train()
                print('error_score:', error_score)
                # 选择误差率小的分类器
                if error_score < best_classifier[0]:
                    best_classifier = (error_score, i, classifier)
            em = best_classifier[0]  # 取出误差率，用来判断是否误差小于阈值
            if em == 0:
                self.alpha.append(100)
            else:
                self.alpha.append(0.5 * math.log((1 - em) / em))  # 更新分类器系数
            self.classifier.append(best_classifier[1:])
            Z = self._Z_(best_classifier[1], best_classifier[2])
            print('alpha:', self.alpha)
            # 计算权值分布
            for i in range(self.N):
                self.w[i] = self._w_(best_classifier[1], best_classifier[2], i) / Z
            print('w:', self.w)

    def _predict_(self, feature):
        result = 0.0
        for i in range(self.M):
            classifier = self.classifier[i][1]
            # 最终的分类器为前面多个分类器的线性组合：G(x) = aG1(x)+aG2(x)+aG3(x)
            result += self.alpha[i] * classifier.predict(feature)
        # 让最终分类器对当前特征打分
        if result > 0:
            return 1
        return -1

    def predict(self, features):
        results = []
        # 把每个特征拿出来计算结果
        for feature in features:
            results.append(self._predict_(feature))
        print('results:', results)
        return results


if __name__ == '__main__':
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    label = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    print('Start Training!')
    ada = AdaBoost()
    ada.train(data, label)
    ada.predict(data)

    # 绘图
    plt.scatter(data, [0.0] * 10, c=label)
    plt.xlim((-1, 11))
    plt.ylim((-2, 2))
    axis_x = np.linspace(0, 9, 10)
    axis_y = ada.predict(data)
    plt.plot(axis_x, axis_y)
    plt.show()
