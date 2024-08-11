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
#df_new.to_csv('F:/fluralaner 208descriptor.csv')

#-----------------------------------------------------------------------------------------------------------------------------
'''
### 测试集描述符提取
selected_columns = ['qed', 'HeavyAtomMolWt', 'ExactMolWt', 'BCUT2D_LOGPLOW', 'Chi0v', 'Kappa3',
                    'LabuteASA', 'SMR_VSA10', 'SMR_VSA3', 'VSA_EState2', 'MolMR','Activity']
predicted_data = df_new[selected_columns]

predicted_data.to_csv('F:/new_px_descriptor.csv')
#'''
