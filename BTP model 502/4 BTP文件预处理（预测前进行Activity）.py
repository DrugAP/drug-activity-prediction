from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
from rdkit.Chem import Descriptors
#mol = Chem.MolFromSmiles('c1ccccc1C(=O)O')

# 1.导入数据
df = pd.read_csv("F:/fluralaner_new_compounds.csv")
mols = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]   

# 2.计算分子的所有描述符
des_list = [x[0] for x in Descriptors._descList]      
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)  
result = [calculator.CalcDescriptors(mols) for mols in mols]  

result_ = [des_list]+result      
result_ = pd.DataFrame(result_[1:], columns=result_[0])   

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
