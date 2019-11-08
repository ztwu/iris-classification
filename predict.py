from sklearn import datasets
from sklearn.linear_model import  LogisticRegression

iris = datasets.load_iris()
X = iris['data']
Y = iris['target']

log_reg = LogisticRegression(multi_class='ovr', solver='sag', max_iter=10000)
log_reg.fit(X, Y)

X_new = [[5.1, 3.5, 1.4, 0.2]]
# predict（X）： 预测X中样本的类标签。
# predict_log_proba（X）： 概率估计的对数。
# predict_proba（X）： 记录概率估计。
print(log_reg.predict_proba(X_new))
print(log_reg.predict_log_proba(X_new))
# print(log_reg._predict_proba_lr(X_new))
print(log_reg.predict(X_new))