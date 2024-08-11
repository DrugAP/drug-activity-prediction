import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier    
from sklearn.metrics import recall_score,precision_score,accuracy_score,roc_auc_score,f1_score,matthews_corrcoef 
from sklearn.model_selection import cross_val_score

# 1.读取训练集和验证集 CSV 文件，获取每个化合物的分子描述符
df = pd.read_csv("F:/Bee_train_14_descriptor_RFE622.csv",index_col = 0)  
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values
print(X.shape)

#-------训练模型----------------------训练模型------------------------训练模型----------------------------------

from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
from sklearn.feature_selection import RFE
cv = StratifiedKFold(n_splits=5)

# 创建一个具有概率估计的随机森林分类器
rf_clf = RandomForestClassifier(n_estimators=110,max_depth=20,min_samples_leaf=1,min_samples_split=2 ,random_state=0)

#--------绘制ROC曲线-----------绘制ROC曲线----------------绘制ROC曲线-----------------------------------------------------

tprs = [] 
aucs = []  
mean_fpr = np.linspace(0, 1, 100) 

fig, ax = plt.subplots()

for i, (train, test) in enumerate(cv.split(X, y)):
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

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0  
    tprs.append(interp_tpr)  
    aucs.append(viz.roc_auc) 

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

# 计算并绘制平均ROC曲线
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0  
mean_auc = auc(mean_fpr, mean_tpr)  
std_auc = np.std(aucs) 
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
ax.legend(loc="lower right") 
plt.savefig('F:/bee ROC.pdf', dpi=300)  
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
#plt.savefig('F:/bee description importance.pdf', dpi=300)   
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
acc_test = accuracy_score(y_test, y_pred_test) 
print("acc_test:{}".format(acc_test))
from sklearn import metrics   
print("RMSE_csj:{}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))))
from sklearn.metrics import matthews_corrcoef  
print("MCC:{}".format(matthews_corrcoef(y_test, y_pred_test)))

#-------预测----------------------预测--------------------------预测--------------------------预测----------------

#'''
# 1.读取测试集 CSV 文件，获取每个化合物的分子描述符
df = pd.read_csv("F:/new_Bee_14_descriptor622.csv",index_col = 0)  

# 2.定义X 、Y
X_cs = df.iloc[:, 0:-1].values
y_cs = df.iloc[:, -1].values
print(X.shape)

y_pred_csj = rf_clf.predict(X_cs)
print(y_pred_csj)
y_proba_pred_csj = rf_clf.predict_proba(X_cs)
print(y_proba_pred_csj)
cs_cm = confusion_matrix(y_cs, y_pred_csj)
print("cs_cm:{}".format(cs_cm))
acc_cs = accuracy_score(y_cs, y_pred_csj) 
print("acc_cs:{}".format(acc_cs))
from sklearn import metrics 
print("RMSE_csj:{}".format(np.sqrt(metrics.mean_squared_error(y_cs, y_pred_csj))))
from sklearn.metrics import matthews_corrcoef 
print("MCC:{}".format(matthews_corrcoef(y_cs, y_pred_csj)))
#'''
