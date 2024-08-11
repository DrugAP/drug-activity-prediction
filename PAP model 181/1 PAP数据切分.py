import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv('F:/fluralaner 208descriptor.csv', index_col=0) 

# 使用train_test_split进行8:2的随机切分
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 提取特征和目标变量
X_train = train_df.drop(columns=['Activity'])
y_train = train_df['Activity']
X_test = test_df.drop(columns=['Activity'])
y_test = test_df['Activity']

# 将训练集和测试集组合回包含目标变量的完整数据框
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# 保存训练集和测试集到新的CSV文件
train_df.to_csv('F:/fluralaner_train_208descriptor.csv', index=True)
test_df.to_csv('F:/fluralaner_test_208descriptor.csv', index=True)
