import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

def load_data():
    """
    加载数据集
    :return:
        X： 花瓣宽度
        Y: 鸢尾花类型
    """
    # 加载sklearn包自带的鸢尾花数据;
    iris = datasets.load_iris()
    # # 查看鸢尾花的数据集
    # print(iris)
    # # 查看鸢尾花的key值；
    # # dict_keys(['data', 'target', 'target_names', 'DESCR','feature_names', 'filename'])
    # print(iris.keys())
    # # 获取鸢尾花的特性： ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # print(iris['feature_names'])
    # print(iris['data'])
    # print(iris['target'])
    # 因为花瓣的相关系数比较高， 所以分类效果比较好， 所以我们就用花瓣宽度当作x;
    # X = iris['data']
    X = iris['data'][:, 3:]
    # 获取分类的结果
    Y = iris['target']
    # print(iris)
    return  X, Y

def configure_plt(plt):
    """
    配置图形的坐标表信息
    """
    # 获取当前的坐标轴, gca = get current axis
    ax = plt.gca()
    # 设置x轴, y周在(0, 0)的位置
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    # 绘制x，y轴说明
    plt.xlabel('petal width (cm)')  # 花瓣宽度
    plt.ylabel('target')    # 鸢尾花类型
    return  plt

def model_train():
    """
    训练模型
    :return:
    """
    # 通过上面的数据做逻辑回归
    """
    multi_class='ovr' : 分类方式； OvR（One vs Rest），一对剩余的意思，有时候也称它为  OvA（One vs All）；一般使用 OvR，更标准；
    solver='sag'，逻辑回归损失函数的优化方法;  sag：即随机平均梯度下降，是梯度下降法的变种
    """
    log_reg = LogisticRegression(multi_class='ovr', solver='sag')
    X, Y = load_data()
    log_reg.fit(X, Y)
    print('w0:', log_reg.coef_)
    print('w1:', log_reg.intercept_)
    return  log_reg

def test_data(log_reg):
    """
    测试数据集
    :param log_reg:
    :return:
    """
    # 创建新的数据集去测试
    #   np.linespace 用于创建等差数列的函数， 会创建一个从0到3的等差数列， 包含1000个值；
    #   reshape生成1000行1列的数组；
    X_new = np.linspace(0, 3, 100).reshape(-1, 1)
    print(X_new)
    # X_new1 = np.hstack((X_new,X_new))
    # X_new1 = np.array([
    #             [1, 2, 1, 2],
    #             [2 ,2, 1, 1]
    #         ])
    # print(X_new1)
    # 概率估计的对数。
    # y_proba = log_reg.predict_log_proba(X_new)
    # print(y_proba)
    # 预测X中样本的类标签
    y_hat = log_reg.predict(X_new)
    # print(y_hat)
    return  X_new, y_hat

def bjhs(log_reg, plt):
    ##  绘制边界函数
    w = log_reg.coef_
    b = log_reg.intercept_
    print("回归系数：", w)
    print("截距：", b)
    # line equation: x0 * w0 + x1 * w1 + b = 0
    ax = [i / 10 for i in range(0, 30)]
    for i, a in enumerate(w):
        ay = a[0] * np.array(ax) + b[i]
        # print(y)
        plt.plot(ax, ay, c='yellow')

def draw_pic():
    """
    绘制图形
    :return:
    """
    X, Y = load_data()
    log_reg = model_train()
    test_X, test_Y = test_data(log_reg)
    import matplotlib.pyplot as plt

    # 边界函数
    bjhs(log_reg,plt)

    plt.scatter(X, Y, c='red')
    # plt.scatter(test_X, test_Y, c='green')
    plt = configure_plt(plt)

    # 显示图
    plt.show()

if __name__ == '__main__':
    draw_pic()
    # load_data()