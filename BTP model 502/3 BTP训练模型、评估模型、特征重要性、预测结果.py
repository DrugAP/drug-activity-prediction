import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier     # 随机森林(Random Forest) BaggingRegressor是随机森林专用
from sklearn.metrics import recall_score,precision_score,accuracy_score,roc_auc_score,f1_score,matthews_corrcoef  # 导入了scikit-learn库中的召回率和ROC曲线面积两个评价指标。
from sklearn.model_selection import cross_val_score

# 1.读取训练集和验证集 CSV 文件，获取每个化合物的分子描述符
df = pd.read_csv("F:/Bee_train_14_descriptor_RFE622.csv",index_col = 0)   # index_col=0——第一列为索引值
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values
print(X.shape)

#-------训练模型----------------------训练模型------------------------训练模型----------------------------------

from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
from sklearn.feature_selection import RFE
# 分类和ROC分析
# 创建具有5个折叠的分层交叉验证对象
cv = StratifiedKFold(n_splits=5)

# 创建一个具有概率估计的随机森林分类器
rf_clf = RandomForestClassifier(n_estimators=110,max_depth=20,min_samples_leaf=1,min_samples_split=2 ,random_state=0)

#--------绘制ROC曲线-----------绘制ROC曲线----------------绘制ROC曲线-----------------------------------------------------

tprs = []  # 用于存储每个折叠的真正率
aucs = []  # 用于存储每个折叠的AUC值
mean_fpr = np.linspace(0, 1, 100)  # 创建均匀分布的点以绘制ROC曲线

# 创建绘图所需的图形和轴
fig, ax = plt.subplots()

# 遍历交叉验证的折叠
for i, (train, test) in enumerate(cv.split(X, y)):
    # 在训练数据上训练分类器
    rf_clf.fit(X[train], y[train])

    # 从训练好的分类器和测试数据生成ROC曲线显示
    viz = RocCurveDisplay.from_estimator(
        rf_clf,
        X[test],
        y[test],
        name="ROC fold {}".format(i+1),
        alpha=0.3,
        lw=1,
        ax=ax,
    )

    # 插值真正率以匹配均值均匀分布的点
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0  # 确保第一个点为（0, 0）
    tprs.append(interp_tpr)  # 存储插值后的真正率
    aucs.append(viz.roc_auc)  # 存储计算得到的AUC值

# 在ROC图上绘制“Chance”线
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

# 计算并绘制平均ROC曲线
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0  # 确保最后一个点为（1, 1）
mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
std_auc = np.std(aucs)  # 计算AUC值的标准差
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

# 计算并绘制平均值 +/- 1个标准差之间的区域
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

# 设置图的限制和标签
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic",
)
ax.legend(loc="lower right")  # 在图上添加图例
plt.savefig('F:/文章一 活性预测/Figure/fig 1 流程图/622 bee ROC曲线.pdf', dpi=300)   # 保存图片到F盘，清晰度dpi
plt.show()

#--------特征重要度-----------特征重要度---------------特征重要度-----------------------------------------------------

features = list(df.columns)
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
num_features = len(importances)
print(range(num_features))
print(indices)
# 将特征重要度以柱状图展示
plt.figure(figsize=(15,10))
plt.title("Feature importances")
plt.bar(range(num_features), importances[indices], color="g", align="center")
plt.xticks(range(num_features), [features[i] for i in indices], rotation='30')
plt.xlim([-1, num_features])
#plt.savefig('F:/文章一 活性预测/Figure/fig S1-15/bee description importance622.pdf', dpi=300)   # 保存图片到F盘，清晰度dpi
plt.show()

# 输出各个特征的重要度
for i in indices:
    print("{0} - {1:.3f}".format(features[i], importances[i]))

#------test----------------------test---------------------test-----------------------test----------------
# 1.导入test
df_test = pd.read_csv("F:/Bee_test_14_descriptor622.csv",index_col = 0)
X_test = df_test.iloc[:, 0:-1].values
y_test = df_test.iloc[:, -1].values
print(X.shape)

y_pred_test = rf_clf.predict(X_test)
print(y_pred_test)
y_proba_pred_test = rf_clf.predict_proba(X_test)
print(y_proba_pred_test)
test_cm = confusion_matrix(y_test, y_pred_test)
print("test_test:{}".format(test_cm))
acc_test = accuracy_score(y_test, y_pred_test)  # 准确率
print("acc_test:{}".format(acc_test))
from sklearn import metrics   # 均方误差RMSE
print("RMSE_csj:{}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))))
from sklearn.metrics import matthews_corrcoef  # 马修斯相关系数
print("MCC:{}".format(matthews_corrcoef(y_test, y_pred_test)))

#-------预测----------------------预测--------------------------预测--------------------------预测----------------

#'''
# 1.读取测试集 CSV 文件，获取每个化合物的分子描述符
df = pd.read_csv("F:/new_Bee_14_descriptor622.csv",index_col = 0)   # index_col=0——第一列为索引值

# 2.定义X 、Y（指纹数据集）将分子描述符转换为NumPy数组，并将其赋给变量X和y。其中，X包含所有指纹，y包含相应的活性标签（二进制分类）
X_cs = df.iloc[:, 0:-1].values
y_cs = df.iloc[:, -1].values
print(X.shape)

y_pred_csj = rf_clf.predict(X_cs)
print(y_pred_csj)
y_proba_pred_csj = rf_clf.predict_proba(X_cs)
print(y_proba_pred_csj)
cs_cm = confusion_matrix(y_cs, y_pred_csj)
print("cs_cm:{}".format(cs_cm))
acc_cs = accuracy_score(y_cs, y_pred_csj)  # 准确率
print("acc_cs:{}".format(acc_cs))
from sklearn import metrics   # 均方误差RMSE
print("RMSE_csj:{}".format(np.sqrt(metrics.mean_squared_error(y_cs, y_pred_csj))))
from sklearn.metrics import matthews_corrcoef  # 马修斯相关系数
print("MCC:{}".format(matthews_corrcoef(y_cs, y_pred_csj)))
#'''
