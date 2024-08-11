import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier    
from sklearn.feature_selection import RFE

# 特征选择：RFE
# 1.导入数据
df = pd.read_csv("F:/fluralaner_train_208descriptor.csv",index_col = 0)
X = df.drop(columns=['Activity'])
y = df['Activity']

# 2.保存原始数据的列名
column_names = df.columns[:-1]

# 3.选择特征
scores_and_params = []
'''
# 确定特征数量
for i in range(60,121,10):
    for j in range(4, 21, 4):
        for k in range(5, 15, 1):
            for u in range(1, 5, 1):
                RFC_ = RandomForestClassifier(n_estimators=i,max_depth=j,min_samples_leaf=u,random_state=0) 
                selector = RFE(RFC_, n_features_to_select=k,step=10).fit(X,y) 

                selected_features_indices = selector.support_ 
                selected_feature_names = X.columns[selected_features_indices]  

                print("Selected feature names: ", selected_feature_names)
                print("n_estimators: {0} \n max_depth: {1} \n n_features_to_select: {2} \n min_samples_leaf: {3}".format(i, j, k, u))

                #print(selector.support_.sum())  
                #print(selector.ranking_)      

                # 4.转换数据
                X_wrapper_features = selector.transform(X)

                # 5.交叉验证
                auc_scores = cross_val_score(RFC_,X_wrapper_features,y,cv=5,scoring='roc_auc').mean()

                print(auc_scores)
                scores_and_params.append((auc_scores, selected_feature_names, i, j, k, u))

scores_and_params_sorted = sorted(scores_and_params, key=lambda x: x[0], reverse=True)
top_10_scores_and_params = scores_and_params_sorted[:10]

for score, feature_names, n_estimators, max_depth, n_features_to_select, min_samples_leaf in top_10_scores_and_params:
    print(f"Score: {score}, Features: {list(feature_names)}, n_estimators: {n_estimators}, "
          f"max_depth: {max_depth}, n_features_to_select: {n_features_to_select}, min_samples_leaf: {min_samples_leaf}")

'''
RFC_ = RandomForestClassifier(n_estimators=90,max_depth=4 ,min_samples_leaf=4,random_state=0) 
selector = RFE(RFC_, n_features_to_select=11,step=10).fit(X,y) 

selected_features_indices = selector.support_ 
selected_feature_names = X.columns[selected_features_indices]  

print("Selected feature names: ", selected_feature_names)

# 4.转换数据
X_wrapper_features = selector.transform(X)

# 5.交叉验证
auc_scores = cross_val_score(RFC_,X_wrapper_features,y,cv=5,scoring='roc_auc').mean()

print(auc_scores)

#---------train描述符提取------------------train描述符提取-----------------------------------------
'''
# 提取指定列名的数据
selected_columns = ['qed', 'HeavyAtomMolWt', 'ExactMolWt', 'BCUT2D_LOGPLOW', 'Chi0v', 'Kappa3',
                    'LabuteASA', 'SMR_VSA10', 'SMR_VSA3', 'VSA_EState2', 'MolMR','Activity']
train_data = df[selected_columns]

#将提取的数据保存到文件中
train_data.to_csv('F:/train_descriptor_RFE.csv')

#---------test描述符提取------------------test描述符提取------------------------------------------------------------------------------

# 1.导入test
df_test = pd.read_csv("F:/fluralaner_test_208descriptor.csv",index_col = 0)

# 提取指定列名的数据
selected_columns = ['qed', 'HeavyAtomMolWt', 'ExactMolWt', 'BCUT2D_LOGPLOW', 'Chi0v', 'Kappa3',
                    'LabuteASA', 'SMR_VSA10', 'SMR_VSA3', 'VSA_EState2', 'MolMR','Activity']
test_data = df_test[selected_columns]

#将提取的数据保存到文件中
test_data.to_csv('F:/111/10PX_test_11_descriptor_RFE622.csv')
#'''
