from numpy import *
from math import *
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号


# create Sigmoid function


def Sigmoid(X):  # X表示待求参数与数据集的乘积
    arr = []
    for X in X:
        arr.append(1.0 / (1 + exp(-X)))
    return mat(arr).T


# 梯度上升函数
def gradAscent(dataMatrix, categoryLabel):
    dataMatrix = mat(dataMatrix)  # 将数据集转化为矩阵形式
    categoryLabel = mat(categoryLabel)  # 将类别标签转化为矩阵形式
    m, n = shape(dataMatrix)
    weights = mat(ones((n, 1)))  # 初始化待求矩阵
    alpha = 0.00001  # 步长
    frequency = 50000  # 迭代次数 迭代次数太少 会影响回归方程性能
    for i in range(frequency):
        multiplier = categoryLabel.transpose() - Sigmoid(dataMatrix * weights)
        weights = weights + alpha * dataMatrix.transpose() * multiplier
    return weights


# 绘图函数
def bestFitCurve(weights, data, label):
    # 好瓜：x_data1 表示密度  y_data1 表示含糖率
    dataArr = array(data)
    n = shape(dataArr)[0]
    x_data1 = []
    y_data1 = []
    x_data2 = []
    y_data2 = []
    for i in range(n):
        if int(label[i]) == 1:
            x_data1.append(dataArr[i][1])
            y_data1.append(dataArr[i][2])
        else:
            x_data2.append(dataArr[i][1])
            y_data2.append(dataArr[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_data1, y_data1, s=30, c='red', marker='s')
    ax.scatter(x_data2, y_data2, s=30, c='green')
    x = arange(0, 1, 0.1)
    y = -(weights[0] + weights[1] * x) / weights[2]  # 令线性回归方程 y=ax0+bx1变换得 0=ax0+bx1+cx2
    ax.plot(x, y)
    plt.title('watermelon3α')
    plt.xlabel("密度")
    plt.ylabel("含糖率")
    plt.show()


if __name__ == '__main__':
    dataMat = []  # 存储数据集
    labelMat = []  # 存储类别标签
    fr = open("C:/Users/ASUS/Desktop/watermelon3α.csv")
    for line in fr.readlines():
        linArr = line.strip().split(",")
        dataMat.append([1, float(linArr[1]), float(linArr[2])])
        labelMat.append(int(linArr[3]))
    bestFitCurve(gradAscent(dataMat, labelMat).getA(), dataMat, labelMat)
