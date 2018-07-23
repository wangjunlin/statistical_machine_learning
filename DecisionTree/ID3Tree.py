"""
实现《统计学习方法》 第5章例题 用的ID3算法
流程参照书上的算法流程
参考https://blog.csdn.net/wds2006sdo/article/details/52849400
参考https://www.jianshu.com/p/3c8e22adf737 
"""

import numpy as np
import math


def create_data():
    """
    年龄：青、中、老：0，1，2
    工作：否、是：0，1
    房子：否、是：0，1
    信贷：一般、好、非常好：0，1，2
    类别：否、是：0，1
    :return: 训练集、标签
    """
    data = [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 1, 1, 0, 1], [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0], [1, 0, 0, 1, 0], [1, 1, 1, 1, 1], [1, 0, 1, 2, 1], [1, 0, 1, 2, 1],
            [2, 0, 1, 2, 1], [2, 0, 1, 1, 1], [2, 1, 0, 1, 1], [2, 1, 0, 2, 1], [2, 0, 0, 0, 0]]
    labels = ['年龄', '有工作', '有自己房子', '信贷情况']
    return data, labels


def calc_ent(train_set):
    """
    # 计算经验熵H(D) = - sum((|Ck|/|D|)log2(|Ck|/|D|)) k = 0~K K为类别，这里只有2类
    :param train_set: 需要计算经验熵的数据集
    :return: 经验熵
    """
    data_entries = len(train_set)  # 整个数据集长度
    label_counts = {}  # 分类类别字典（类别：数量）
    for feature in train_set:
        current_label = feature[-1]  # 读取一条数据的最后一个值，统计属于k类的样本个数Ck（这里有2类，是和否）
        # 如果该类别不存在于字典中，则添加，如果已存在，则数量+1
        if current_label not in label_counts.keys():
            label_counts.setdefault(current_label, 0)
        label_counts[current_label] += 1

    ent = 0.0  # 经验熵
    for key in label_counts:
        p = float(label_counts[key]) / data_entries  # p=|Ck|/|D|
        ent -= p * math.log2(p)  # H(D)= -p*log2(p)
    return ent


def calc_condition_ent(train_set, feature):
    """
    # 计算经验条件熵H(D|A) = sum((|Di|/|D|)*H(Di))，i=i~n，n为特征数量
    :param train_set: 数据集
    :param feature: 特征
    :return: 经验条件熵
    """
    total_num = len(train_set)  # |D|
    feature_value = set(data[feature] for data in train_set)  # 该特征取值的集合
    condition_ent = 0.0
    # 从特征的取值集合遍历符合取值的数据集
    for value in feature_value:
        feature_set = [data for data in train_set if data[feature] == value]  # |Di|
        p = len(feature_set) / float(total_num)  # |Di|/|D|
        condition_ent += p * calc_ent(feature_set)  # H(Di)
    return condition_ent


def calc_gain(ent, condition_ent):
    """
    # 计算信息增益
    :param ent: 经验熵
    :param condition_ent: 经验条件熵
    :return: 信息增益
    """
    return ent - condition_ent


# 获取最大信息增益的特征
def get_best_feature(train_set, labels):
    best_gain = 0
    best_feature = -1
    feature_num = len(train_set[0]) - 1  # 获取除了最后一列的特征数量
    ent = calc_ent(train_set)  # 计算训练集经验熵
    # 遍历特征:0,1,2...
    for feature in range(feature_num):
        condition_ent = calc_condition_ent(train_set, feature)
        gain = calc_gain(ent, condition_ent)
        print('feature {}, gain {}'.format(feature, gain))
        # 找出最大的信息增益
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
    return best_feature, labels[best_feature], best_gain


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count.setdefault(vote, 0)
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=lambda i: i[1], reverse=True)
    return sorted_class_count[0][0]


# 输入三个变量（带划分数据集， 特征，分类值)
def split_data_set(data, axis, value):
    ret_data_set = []
    for feat_vec in data:
        if feat_vec[axis] == value:
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduce_feat_vec)
    return ret_data_set  # 返回不含划分特征的子集


def create_tree(train_set, train_labels):
    # 分类类别list
    class_list = [line[-1] for line in train_set]
    # 当所有实例同属一类，Tree为单节点树，将类Ck作为该节点的类标记，返回Tree
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 当遍历完特征，返回实例数最大的类作为节点标记
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    # 按照信息增益最大选择特征A
    best_feature, best_label, best_gain = get_best_feature(train_set, train_labels)

    my_tree = {best_label: {}}

    del (train_labels[best_feature])
    feature_values = [line[best_feature] for line in data_set]
    # 特征取值集合
    unique_values = set(feature_values)
    for value in unique_values:
        sub_label = labels[:]
        my_tree[best_label][value] = create_tree(split_data_set(train_set, best_feature, value), sub_label)

    return my_tree


if __name__ == '__main__':
    data_set, labels = create_data()
    # print(calc_ent(data))  # 0.971，符合书上例题的计算结果
    # print(get_best_feature(data_set, label))  # feature == 2时信息增益最大（有自己房子），符合计算结果
    tree = create_tree(data_set, labels)
    print('tree->', tree)
