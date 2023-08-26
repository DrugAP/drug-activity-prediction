import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
'''
# 特征选择：包装法wrapper
from sklearn.feature_selection import RFE
# 1.导入数据
df = pd.read_csv("F:/20230812 训练集177 mol descriptor全部.csv",index_col = 0)

X = df.iloc[:, 0:-1]                            # 将每个分子的指纹信息转换为 numpy 数组；[:, 0:-1]左闭右开不取最后一列
print(X.shape)
y = np.array(df.Activity)                       # 将分子活性信息转换为 numpy 数组
#print(y)

# 2.保存原始数据的列名
column_names = df.columns[:-1]

# 3.选择特征
#for i in range(6,16,1):
RFC_ = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=3,random_state=0)  # 随机森林实例化
selector = RFE(RFC_, n_features_to_select=12,step=10).fit(X,y)  # RFE实例化

selected_features_indices = selector.support_  # 返回所有的特征是否被选中的布尔矩阵
selected_feature_names = X.columns[selected_features_indices]  # 选中特征的列名

print("Selected feature names: ", selected_feature_names)
#print(selector.support_.sum())    # support_:返回所有的特征是否被选中的布尔矩阵,求和后得到30个特征
#print(selector.ranking_)        # ranking_:返回特征的按数次迭代中综合重要性的排名 排第1的特征最重要

# 4.转换数据
X_wrapper_features = selector.transform(X)

# 5.交叉验证
print(cross_val_score(RFC_,X_wrapper_features,y,cv=5,scoring='roc_auc').mean())

# 6.提取保留的特征列
selected_columns = column_names[selector.get_support()]  # get_support()这个方法必须直接使用在选择器上
# 7.将保留的特征列转换为 DataFrame
selected_X = pd.DataFrame(X_wrapper_features, columns=selected_columns)
# 8.提取原始df的'Activity'列
activity_column = df['Activity']
# 9.将最后一列添加到新 DataFrame
df_new = pd.concat([selected_X, activity_column], axis=1)

# 10.保存到csv
#pd.DataFrame(df_new).to_csv(r'F:/20230812 177 mol descriptor包装法wrapper.csv')
'''
#-----------------------------------------------------------------------------------------------------------


# 特征间相关性分析
import seaborn as sns

# 1.导入数据
df = pd.read_csv("F:/20230812 177 mol descriptor包装法wrapper-0.83077.csv",index_col = 0)

X = df.iloc[:, 0:-1].values                              # 将每个分子的指纹信息转换为 numpy 数组；[:, 0:-1]左闭右开不取最后一列
print(X.shape)
# y = np.array(df.Activity)                                        # 将分子活性信息转换为 numpy 数组
# print(y)

# 2.计算pearson相关性系数
X_pearson = df.corr(method='pearson', min_periods=1)
print(X_pearson)

# 3.保存到csv
#pd.DataFrame(X_pearson).to_csv(r'F:\20230613 172 mol descriptor pearson相关性分析结果.csv')

# 4.画热图并保存图片
plt.figure(figsize=(30,30))
sns.heatmap(df.corr(),annot=True)    # cmap属性修改颜色
plt.savefig('F:/20230614heatmap pearson相关性分析.pdf', dpi=700)   # 保存图片到F盘，清晰度dpi
plt.show()

