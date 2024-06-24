import pandas as pd
from sklearn.model_selection import train_test_split

# 假设数据保存在一个名为'data.csv'的文件中
# 读取数据
df = pd.read_csv('F:/bees 208descriptor 240409.csv', index_col=0) #index_col=0 保留行名和列名

# 使用train_test_split进行8:2的随机切分
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 提取特征和目标变量
X_train = train_df.drop(columns=['Toxicity'])
y_train = train_df['Toxicity']
X_test = test_df.drop(columns=['Toxicity'])
y_test = test_df['Toxicity']

# 将训练集和测试集组合回包含目标变量的完整数据框
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# 保存训练集和测试集到新的CSV文件
train_df.to_csv('F:/Bee_train_208descriptor.csv', index=True)
test_df.to_csv('F:/Bee_test_208descriptor.csv', index=True)
