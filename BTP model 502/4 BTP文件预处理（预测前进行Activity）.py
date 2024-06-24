 ### 测试集描述符计算  从SMILES字符串直接拿到可以用于预测的文件（有描述符）
# 计算分子描述符https://blog.csdn.net/dreadlesss/article/details/106163464?ops_request_misc=&request_id=&biz_id=102&utm_term=rdkit%E8%AE%A1%E7%AE%97%E6%89%80%E6%9C%89%E5%88%86%E5%AD%90%E6%8F%8F%E8%BF%B0%E7%AC%A6&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-106163464.142^v81^insert_down38,201^v4^add_ask,239^v2^insert_chatgpt&spm=1018.2226.3001.4187
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
from rdkit.Chem import Descriptors
#mol = Chem.MolFromSmiles('c1ccccc1C(=O)O')

# 1.导入数据
df = pd.read_csv("F:/fluralaner_new_compounds.csv")
mols = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]   # 将 SMILES 字符串转换为分子结构，转换为RDKit中的Mol对象，添加为数据框df中的一列

# 2.计算分子的所有描述符
des_list = [x[0] for x in Descriptors._descList]       # 获取所有描述符：Descriptors._descList  存到des_list
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)  # 批量计算描述符
result = [calculator.CalcDescriptors(mols) for mols in mols]   # 批量计算mols中每个化合物的描述符  result类型是列表套元祖

result_ = [des_list]+result      # des_list一维，result_二维，合并两个列表得到新的二维列表result_
result_ = pd.DataFrame(result_[1:], columns=result_[0])    # Pandas将列表（List）转换为数据框（Dataframe） (c[1:], columns=c[0])使用第一行作为列标签并删除原始数据中的第一行（列标签）

# 3.将原始df的'Activity'列添加到df_new
#smi_column = df['SMILES']
activity_column = df['Activity']
df_new = pd.concat([result_, activity_column], axis=1)
#df_new.to_csv('F:/fluralaner 208descriptor 240409.csv')

#-----------------------------------------------------------------------------------------------------------------------------
'''
### 测试集描述符提取
# 提取指定列名的数据
selected_columns = ['qed', 'MaxPartialCharge', 'MinPartialCharge', 'BCUT2D_MWLOW', 'BCUT2D_LOGPHI', 'Chi2n', 
                    'Chi3n', 'Chi3v', 'Chi4n', 'HallKierAlpha', 'SMR_VSA7'   ,'Activity']
predicted_data = df_new[selected_columns]

#将提取的数据保存到文件中
predicted_data.to_csv('F:/new/10new_Bee_11_descriptor622.csv')
#'''