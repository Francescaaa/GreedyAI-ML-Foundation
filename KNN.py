from sklearn import datasets
from collections import Counter  # 做投票
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target
# random_state 是类似seed的一种东西
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)


def euc_dis(instance1, instance2):
    dist = np.sqrt(sum((instance1 - instance2)**2))
    return dist


def knn_classify(X, y, testInstance, k):
    distances = [euc_dis(x, testInstance) for x in X]
    kneighbors = np.argsort(distances)[:k]
    count = Counter(y[kneighbors])
    return count.most_common()[0][0]


predictions = [knn_classify(X_train, y_train, data, 3)] for data in X_test]

# 或者使用Sklearn自带的工具
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)

correct = np.count_nonzero((predictions == y_test) == True)
print("Accuracy is: %.3f" % (correct/len(X_test)))

# K值对决策边界的影响
from itertools import product
# 用在可视化模块
from sklearn.neighnors import KNeighborsClassifier

# 生成一些随机样本
n_points = 100
X1= np.random.multivariate_normal([1, 50], [[1, 0], [0, 10]], n_points)
X1= np.random.multivariate_normal([2, 50], [[1, 0], [0, 10]], n_points)
X= np.concatenate([X1, X2])
y = np.array([0]*n_points + [1]*n_points)
print(X.shape, y.shape)

# 训练9个K值不同的KNN
clf = []
neighbors = [1, 3, 5, 9, 11, 13, 15, 17, 19]
for i in range(len(neighbors)):
    clfs.append(KNeighborsClassifier(n_neighbors=neighbors[i]).fit(X, y))

# 可视化结果
x_min, x_max = X[:, 0]
# TODO:

# Cross-validation, k = N
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X.shape, y.shape)

ks= [1, 3, 5, 7, 9, 11, 13, 15]
kf= KFold(n_splits = 5, random_state = 2001, shuffle=True)

best_k = ks[0]
best_score = 0

for k in ks:
    curr_score = 0
    for train_index, valid_index in kf.split(X):
        clf = KNeighborsClassifier(n_neighbors = k)
        clf.fit(X[train_index], y[train_index])
        curr_score = curr_score + clf.score(X[valid_index], y[valid_index])
    avg_score = curr_score/5
    if avg_score > best_score:
        best_k = k
        best_score = avg_score
    print("current best score is: %.2f" % best_score, "best k: %d" % best_k)

print("after cross validation, the final best k is: %d" % best_k)

# 或者使用Sklearn来实现
from sklearn.model_selection import GridSearchCV  # 网格方式来搜索参数
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

parameters = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}
knn = KNeighborsClassifier()

clf = GridSearchCV(knn, parameters, cv = 5)
clf.fit(X, y)

print("best score is: %.2f" % clf.best_score_, "  best param: ", clf.best_params_)
