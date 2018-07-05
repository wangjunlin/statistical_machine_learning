import numpy as np
import matplotlib.pyplot as plt
import random

# 对《统计学习方法》31页例题的代码实现
# 参考 https://blog.csdn.net/W_peijian/article/details/79098649

# 数据点
x = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, -1])

# 绘点
plt.plot(x[0][0], x[0][1], 'ro')
plt.plot(x[1][0], x[1][1], 'ro')
plt.plot(x[2][0], x[2][1], 'rx')

# 初始化权重、偏置、学习率
w = np.zeros((1, 2))
b = 0
study_rate = 1

random.seed(100)
epochs = 100

for epoch in range(epochs):
    # 对于每个epoch，随机选取一个数据xi yi，判断 yi(np.dot(w,xi)+b)是否小于0
    # 如果小于等于0则是误分类点（未正确分类）
    index = random.randint(0, 2)
    result = y[index] * (np.dot(w, x[index]) + b)
    if result <= 0:
        # w,b更新
        w = w + study_rate * y[index] * x[index]
        b = b + study_rate * y[index]

print('w: {} \t b: {}'.format(w, b))

line_x = [0, 6]
line_y = [0, 0]

# 不同的初值和不同的误分类点顺序会使解不一样
for i in range(len(line_x)):
    line_y[i] = (-w[0][0] * line_x[i] - b) / w[0][1]

plt.plot(line_x, line_y)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()
