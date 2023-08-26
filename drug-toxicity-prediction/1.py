import pandas as pd
import numpy as np
#from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler  # 导入了imbalanced-learn库中的随机下采样工具，用于处理数据集中的类别不平衡问题。
from sklearn.metrics import recall_score,precision_score,accuracy_score,roc_auc_score,f1_score,matthews_corrcoef,roc_curve  # 导入了scikit-learn库中的召回率和ROC曲线面积两个评价指标。
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.model_selection import train_test_split    # 导入了scikit-learn库中的交叉验证工具，包括K折交叉验证、分层K折交叉验证、分层随机洗牌交叉验证和训练集-测试集划分工具
# from matplotlib import cm                               # 导入了matplotlib库中的颜色映射模块，用于可视化


from sklearn.linear_model import LogisticRegression               # 逻辑回归(Logistic Regression)
from sklearn.tree import DecisionTreeClassifier                   # 决策树(Decision Tree)
from sklearn.ensemble import RandomForestClassifier               # 随机森林(Random Forest) BaggingRegressor是随机森林专用
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVC                                       # 支持向量机(Support Vector Machine, SVM)
# from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB                        # 朴素贝叶斯(Naive Bayes)
# from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier                # K最近邻(K-Nearest Neighbor, KNN)
from sklearn.neural_network import MLPClassifier                  # 神经网络(Neural Network)

# 5.读取训练集和验证集 CSV 文件，获取每个化合物的分子描述符
df = pd.read_csv("F:/20230614 163 mol descriptor逐步回归余10.csv",index_col = 0)          # index_col=0——第一列为索引值

# 6.定义X 、Y（指纹数据集）将分子描述符转换为NumPy数组，并将其赋给变量X和y。其中，X包含所有指纹，y包含相应的活性标签（二进制分类）
X = df.iloc[:, 0:-1].values                              # 将每个分子的指纹信息转换为 numpy 数组
print(X.shape)
y = df.iloc[:, -1].values                             # 将分子活性信息转换为 numpy 数组
print(y.shape)

# 7.读取测试集 CSV 文件，获取每个化合物的分子描述符
df_cs = pd.read_csv("F:/20230614 163 mol descriptor_test.csv",index_col = 0)          # index_col=0——第一列为索引值

# 6.定义X 、Y（指纹数据集）将分子描述符转换为NumPy数组，并将其赋给变量X和y。其中，X包含所有指纹，y包含相应的活性标签（二进制分类）
X_cs = df_cs.iloc[:, 0:-1].values                              # 将每个分子的指纹信息转换为 numpy 数组
print(X_cs.shape)
y_cs = df_cs.iloc[:, -1].values                             # 将分子活性信息转换为 numpy 数组
print(y_cs.shape)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X , y,test_size=0.3)  # 将训练集和测试集从X和y中提取出来

# 交叉验证
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# 混淆矩阵
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# 均方误差RMSE
from sklearn import metrics

### n_estimators的学习曲线-混淆矩阵
cm_xlj = np.array([])         # 训练集混淆矩阵
cm = np.array([])             # 验证集混淆矩阵
cm_csj = np.array([])         # 测试集混淆矩阵
auc  = np.array([])
acc_yzj = np.array([])
acc_xlj = np.array([])
acc_csj = np.array([])


rf_clf_cm = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_leaf=1)       # 实例化
rf_clf_cm = rf_clf_cm.fit(X_train, y_train)                # 训练模型
rf_clf_s = cross_val_score(rf_clf_cm, X, y, cv=10,scoring='roc_auc')            # 交叉验证

y_pred_rf_cm = rf_clf_cm.predict(X_test)                   # 预测验证集
cm_ = confusion_matrix(y_test, y_pred_rf_cm)
cm_ = np.array(cm_)                                        # 将列表cm转换为NumPy数组
cm = np.append(cm, cm_)
acc_yzj = np.append(acc_yzj, accuracy_score(y_test, y_pred_rf_cm))


y_pred_xlj= rf_clf_cm.predict(X_train)                      # 预测训练集
cm_xlj_ = confusion_matrix(y_train, y_pred_xlj)
cm_xlj_ = np.array(cm_xlj_)                                 # 将列表cm_xlj转换为NumPy数组
cm_xlj = np.append(cm_xlj, cm_xlj_)
acc_xlj = np.append(acc_xlj, accuracy_score(y_train, y_pred_xlj))

y_pred_csj = rf_clf_cm.predict(X_cs)                        # 预测测试集
cm_csj_ = confusion_matrix(y_cs, y_pred_csj)
cm_csj_ = np.array(cm_csj_)                                 # 将列表cm_csj转换为NumPy数组
cm_csj = np.append(cm_csj, cm_csj_)
acc_csj = np.append(acc_csj, accuracy_score(y_cs, y_pred_csj))


# 计算train和validation的auc值并绘图
y_pred_proba = rf_clf_cm.predict_proba(X)[:, 1]
auc_X = roc_auc_score(y, y_pred_proba)
print(auc_X)

# 计算FPR，recall,thresholds
FPR_X,recall_X,thresholds_X = roc_curve(y,y_pred_proba,pos_label=1)
area_X = roc_auc_score(y, y_pred_proba)

# 绘制ROC曲线
plt.figure()
plt.plot(FPR_X,recall_X,color='red',label='ROC curve(area = %0.2f)' % area_X)
plt.plot([0,1],[0,1],color='black',linestyle='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic of X')
plt.legend(loc='lower right')
plt.show()

# 计算test的auc值并绘图
y_pred_proba = rf_clf_cm.predict_proba(X_cs)[:, 1]
auc_cs = roc_auc_score(y_cs, y_pred_proba)
print(auc_cs)

# 计算FPR，recall,thresholds
FPR_cs,recall_cs,thresholds_cs = roc_curve(y_cs,y_pred_proba,pos_label=1)
area_cs = roc_auc_score(y_cs, y_pred_proba)

# 绘制ROC曲线
plt.figure()
plt.plot(FPR_cs,recall_cs,color='red',label='ROC curve(area = %0.2f)' % area_cs)
plt.plot([0,1],[0,1],color='black',linestyle='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic of cs')
plt.legend(loc='lower right')
plt.show()


np.set_printoptions(linewidth=18)
print(cm_xlj)
print(cm)
print(cm_csj)
print('acc_xlj:\t%.2f +/- %.2f' % (acc_xlj.mean(),acc_xlj.std()))
print('acc_yzj:\t%.2f +/- %.2f' % (acc_yzj.mean(),acc_yzj.std()))
print('acc_csj:\t%.2f +/- %.2f' % (acc_csj.mean(),acc_csj.std()))

# 特征重要性
print("feature importance:\n{}".format(rf_clf_cm.feature_importances_))

# RMSE
print("RMSE_xlj:\n{}".format(np.sqrt(metrics.mean_squared_error(y_train, y_pred_xlj))))
print("RMSE_yzj:\n{}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf_cm))))
print("RMSE_csj:\n{}".format(np.sqrt(metrics.mean_squared_error(y_cs, y_pred_csj))))

# 交叉验证结果，输出值为auc
print(rf_clf_s)
print(rf_clf_s.mean())

