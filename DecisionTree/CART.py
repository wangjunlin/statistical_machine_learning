"""
cart分类树
参考https://blog.csdn.net/WiseDoge/article/details/57077787
"""


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


def calc_gini(train_data, feature_index, features_value_set, labels_value_set):
    """
    得到给定特征索引下的最优切分点a和切分点a的基尼指数
    :param train_data:训练集字集 
    :param feature_index:
    :param features_value_set: 
    :param labels_value_set: 
    :return:该特征索引的特征值，该特征值的基尼指数 
    """
    gini = 10000
    feature_value_index = 0
    data_length = len(train_data)  # |D|
    # 对每个特征的取值进行计算gini指数
    for feature_value in features_value_set:
        feature_length = features_value_set[feature_value]  # |D1|
        other_feature_length = data_length - feature_length  # |D2| = |D| - |D1|
        feature_label_set = {}  # 每个子特征的分类标签集合
        for data in train_data:
            if data[feature_index] == feature_value:
                if data[-1] not in feature_label_set.keys():
                    feature_label_set.setdefault(data[-1], 0)
                feature_label_set[data[-1]] += 1
        for feature_label_value in feature_label_set.keys():
            p1 = feature_label_set[feature_label_value] / feature_length
            p2 = (labels_value_set[feature_label_value] - feature_label_set[feature_label_value]) / other_feature_length
            break
        # gini(D,A) = |D1|/|D|gini(D1)+|D2|/|D|gini(D2)
        count = ((feature_length / data_length) * 2 * p1 * (1 - p1)) + (
            (other_feature_length / data_length) * 2 * p2 * (1 - p2))
        # 比较基尼指数，取较小值
        if count <= gini:
            gini = count
            feature_value_index = feature_value
    return feature_value_index, gini


def get_best_feature(train_data):
    feature_length = len(train_data[0]) - 1  # 特征个数
    best_gini = 10000
    best_feature = 0
    best_feature_value = 0
    # 获得标签取值集合
    labels_value_set = {}  # 取值：数量
    for data in train_data:
        current_index = data[-1]
        if current_index not in labels_value_set.keys():
            labels_value_set.setdefault(current_index, 0)
        labels_value_set[current_index] += 1
    # 获得特征标签集合,并计算GINI指数
    for feature_index in range(feature_length):
        features_value_set = {}
        for data in train_data:
            feature_value_index = data[feature_index]
            if feature_value_index not in features_value_set.keys():
                features_value_set.setdefault(feature_value_index, 0)
            features_value_set[feature_value_index] += 1
        temp_feature_value, temp_gini = calc_gini(train_data, feature_index, features_value_set, labels_value_set)
        print('feature:', feature_index, 'feature_value:', temp_feature_value, 'gini:', temp_gini)
        # 每次比较特征的gini指数，取最小的gini指数
        if temp_gini < best_gini:
            best_gini = temp_gini
            best_feature = feature_index
            best_feature_value = temp_feature_value
    print('best_feature:{},best_feature_value:{},best_gini:{}'.format(labels[best_feature], best_feature_value,
                                                                      best_gini))
    return best_feature, best_feature_value


def split_data_set(data, feature, value):
    """
    根据给定特征切分数据集，返回不含该特征列的数据集
    :param data: 待切分数据集
    :param feature: 指定特征
    :param value: 指定特征的值
    :return: 不带指定特征的数据集
    """
    ret_data_set = []
    for feat_vec in data:
        if feat_vec[feature] == value:
            reduce_feat_vec = feat_vec[:feature]
            reduce_feat_vec.extend(feat_vec[feature + 1:])
            ret_data_set.append(reduce_feat_vec)
    return ret_data_set


# 辅助函数，投票表决，返回数量最多的类别
def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count.setdefault(vote, 0)
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=lambda i: i[1], reverse=True)
    return sorted_class_count[0][0]


def build_tree(train_data, train_label):
    """
    构建二叉树，同ID3算法，只是改变把信息增益换成了GINI指数
    :param train_data: 训练集
    :param train_label: 标签
    :return: 
    """
    targets = [target[-1] for target in train_data]
    if len(set(targets)) == 1:
        return targets[0]
    if len(train_data[0]) == 1:
        return majority_cnt(targets)

    best_feature_index, best_feature_value = get_best_feature(train_data)
    temp_label = train_label.copy()
    best_feature_label = temp_label[best_feature_index]
    del temp_label[best_feature_index]
    my_tree = {best_feature_label: {}}
    feature_values = [line[best_feature_index] for line in data_set]
    # 特征取值集合
    unique_values = set(feature_values)
    for value in unique_values:
        sub_label = labels[:]
        my_tree[best_feature_label][value] = build_tree(split_data_set(train_data, best_feature_index, value),
                                                        sub_label)
    return my_tree

if __name__ == '__main__':
    data_set, labels = create_data()
    # print(get_best_feature(data_set))  # 最好切分特征为有自己房子，切分点为1
    print(build_tree(data_set, labels))
