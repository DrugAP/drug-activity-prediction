import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    # 导入了scikit-learn库中的交叉验证工具，包括K折交叉验证、分层K折交叉验证、分层随机洗牌交叉验证和训练集-测试集划分工具

from sklearn.ensemble import RandomForestClassifier     # 随机森林(Random Forest) BaggingRegressor是随机森林专用
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import plot_confusion_matrix, precision_score, recall_score,f1_score,accuracy_score,roc_auc_score
from sklearn.model_selection import cross_val_score

# 1.读取训练集和验证集 CSV 文件，获取每个化合物的分子描述符
df = pd.read_csv("F:/20230812 177 mol descriptor包装法wrapper-0.83077.csv",index_col = 0)   # index_col=0——第一列为索引值

# 2.定义X 、Y（指纹数据集）将分子描述符转换为NumPy数组，并将其赋给变量X和y。其中，X包含所有指纹，y包含相应的活性标签（二进制分类）
X = df.iloc[:, 0:-1].values                                              # 将每个分子的指纹信息转换为 numpy 数组
print(X.shape)
y = df.iloc[:, -1].values                                                # 将分子活性信息转换为 numpy 数组
#print(y.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=0)  # 将训练集和测试集从X和y中提取出来

#-------调-参-----------------------调-参-----------------------调-参-----------------------------------------------------
# GridSearchCV（网格搜索交叉验证）：
clf_rf_grid = RandomForestClassifier(n_estimators=100)
parameters = {'max_depth' : range(5,20),
              'min_samples_split' : range(1,8)}
grid_search_cv_clf_rf = GridSearchCV(clf_rf_grid, parameters, cv=5)

# 训练模型
grid_search_cv_clf_rf.fit(X_train,y_train)

# 保存模型
best_model_grid = grid_search_cv_clf_rf.best_estimator_

 # 模型在测试集上的正确率
print("Test set score: {:.2f}".format(grid_search_cv_clf_rf.score(X_test, y_test)))
# 最佳正确率下的模型参数，也就是param_grid中的组合
print("Best parameters: {}".format(grid_search_cv_clf_rf.best_params_))
# 模型调参时的最佳正确率
print("Best cross-validation score: {:.2f}".format(grid_search_cv_clf_rf.best_score_))
# 最佳模型
print("Best estimator:\n{}".format(grid_search_cv_clf_rf.best_estimator_))


accuracy = best_model_grid.score(X_test, y_test)
y_pred_proba = best_model_grid.predict_proba(X_test)[:, 1]
auc_X_test = roc_auc_score(y_test, y_pred_proba)
y_test_predicted = best_model_grid.predict(X_test)
precision = precision_score(y_test, y_test_predicted)
recall = recall_score(y_test, y_test_predicted)
f1score = f1_score(y_test, y_test_predicted)
print('Accuracy: {0} \n AUC score: {1} \n Precision score: {2} \n Recall score: {3} \n F1-score: {4}'\
          .format(accuracy, auc_X_test, precision, recall, f1score))