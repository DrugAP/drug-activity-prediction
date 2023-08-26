### 测试集描述符计算  从SMILES字符串直接拿到可以用于预测的文件（有描述符）
# 计算分子描述符https://blog.csdn.net/dreadlesss/article/details/106163464?ops_request_misc=&request_id=&biz_id=102&utm_term=rdkit%E8%AE%A1%E7%AE%97%E6%89%80%E6%9C%89%E5%88%86%E5%AD%90%E6%8F%8F%E8%BF%B0%E7%AC%A6&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-106163464.142^v81^insert_down38,201^v4^add_ask,239^v2^insert_chatgpt&spm=1018.2226.3001.4187
from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
#mol = Chem.MolFromSmiles('c1ccccc1C(=O)O')

from rdkit.ML.Descriptors import MoleculeDescriptors
des_list = [x[0] for x in Descriptors._descList]       # 获取所有描述符：Descriptors._descList  存到des_list
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)  # 批量计算描述符

# 显示全部输出结果
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# 1.导入数据
df = pd.read_csv("F:/氟雷拉纳1.csv")
PandasTools.AddMoleculeColumnToFrame(df,'SMILES','mol',includeFingerprints=True)
df['MW'] = df['mol'].apply(Descriptors.MolWt)

X = [smiles for smiles in df['SMILES']]
aa = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]   # 将 SMILES 字符串转换为分子结构，转换为RDKit中的Mol对象，添加为数据框df中的一列
# print(X)   打印所有的smiles字符串
# print(aa)  打印所有SMILES 字符串转换过来的分子结构，转换为RDKit中的Mol对象

# 2.计算分子的所有描述符
ab = [calculator.CalcDescriptors(aa) for aa in aa]   # 批量计算aa中每个化合物的描述符  ab类型是列表套元祖
new_ab = list(map(list, ab))   # 使用map()函数将ab中的每个元组转换为一个列表，并将结果收集到new_ab列表中。需要使用 list() 函数将map()的输出转换为列表。
c = [des_list]+new_ab      # des_list一维，new_ab二维，合并两个列表得到新的二维列表c

df1 = pd.DataFrame(c[1:], columns=c[0])    # Pandas将列表（List）转换为数据框（Dataframe） (c[1:], columns=c[0])使用第一行作为列标签并删除原始数据中的第一行（列标签）
#print(df1)

# 3.提取原始df的'Activity'列
activity_column = df['Activity']

# 4.将最后一列添加到新 DataFrame
df_new = pd.concat([df1, activity_column], axis=1)

#-----------------------------------------------------------------------------------------------------------------------------

### 测试集描述符提取
# 提取指定列名的数据
selected_columns = ['qed', 'ExactMolWt', 'FpDensityMorgan2', 'BCUT2D_LOGPLOW',
       'BCUT2D_MRHI', 'Chi0v', 'Chi2v', 'Kappa3', 'LabuteASA', 'SMR_VSA10',
       'SMR_VSA3', 'VSA_EState2', 'Activity']
predicted_data = df_new[selected_columns]

# 将提取的数据保存到文件中
predicted_data.to_csv('F:/20r.csv')
