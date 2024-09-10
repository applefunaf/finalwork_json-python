import os
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from enum import IntEnum
import sys, glob, os, math, numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
ncomponents=100

class Role(IntEnum):
    REACTANT = 0
    REAGENT = 1
    CATALYST = 2
    SOLVENT = 3
    

@dataclass
class Amount:
    unit: str
    value: float

@dataclass
class Material:
    name: str
    smiles: str
    amount: Amount
    role: Role
    def molecule(self):
        return Chem.MolFromSmiles(self.smiles)

_mgen = GetMorganGenerator(radius=2)
def fingerprint_of(molecule):
    return list(_mgen.GetFingerprint(molecule))

@dataclass
class Product:
    smiles: str
    desired: bool
    percent: float
def outcome_as_product(data: dict) -> Product:
    smiles = data['identifiers'][0]['value']
    percent = data['measurements'][0]['percentage']['value']
    desired = data['is_desired_product']
    return Product(smiles, desired, percent)


def input_as_material(name: str, data: dict) -> tuple:
    # 处理第一个组件
    component1 = data['components'][0]
    smiles1 = component1['identifiers'][0]['value']
    try:
        amount1 = Amount('MOLE', component1['amount']['moles']['value'])
    except KeyError:
        amount1 = Amount('LITER', component1['amount']['volume']['value'])
    role1 = getattr(Role, component1['reaction_role'])
    material1 = Material(name + "1", smiles1, amount1, role1)

    # 处理第二个组件，如果存在
    material2 = None
    if len(data['components']) > 1:
        component2 = data['components'][1]
        smiles2 = component2['identifiers'][0]['value']
        try:
            amount2 = Amount('MOLE', component2['amount']['moles']['value'])
        except KeyError:
            amount2 = Amount('LITER', component2['amount']['volume']['value'])
        role2 = getattr(Role, component2['reaction_role'])
        material2 = Material(name + "2", smiles2, amount2, role2)
   

    return material1, material2


# 指定包含 JSON 文件的本地文件夹路径
folder_path = "C:\\User\\35078\\OneDrive\\桌面\\大二上\\暑期程设\\课堂\\大作业\\00"

keys = set()

if '--cache' not in sys.argv:
    data_list = []

    for filename in glob.iglob(os.path.join(folder_path, '*.json')):
        data = json.load(open(filename))
        inputs = [input_as_material(name, inp) for name, inp in data['inputs'].items()]
        d = {
            'temperature': data['conditions']['temperature']['setpoint']['value'],
            #'Solvent': 0
        }
        data_list.append(d)
        for material_pair in inputs:
            # 解包每个元组中的 Material 对象
            material1, material2 = material_pair
            d[material1.name] = material1.amount.value
            if material2:
                d[material2.name] = material2.amount.value
            if 1:
                molecule1 = material1.molecule()
                for name, desc in Descriptors.descList:
                    d[f'{material1.name}::{name}'] = desc(molecule1)
                for index, fg in enumerate(fingerprint_of(molecule1)):
                    d[f'{material1.name}::fingerprint_{index}'] = fg
            if material2 :
                molecule2 = material2.molecule()
                for name, desc in Descriptors.descList:
                    d[f'{material2.name}::{name}'] = desc(molecule2)
                for index, fg in enumerate(fingerprint_of(molecule2)):
                    d[f'{material2.name}::fingerprint_{index}'] = fg
            
                    
        product = outcome_as_product(data['outcomes'][0]['products'][0])
        d['product'] = product.percent
        keys |= d.keys()

    avg_temp = np.average([data['temperature'] for data in data_list if isinstance(data['temperature'], (int, float))])
    for row in data_list:
        if not isinstance(row['temperature'], (int, float)):
            row['temperature'] = avg_temp
        for key in keys:
            if key not in row or math.isnan(row[key]):
                row[key] = 0

    df = pd.DataFrame(data_list)
    for column in df.columns:
        zero_count = (df[column] == 0).sum()
        if zero_count > 730  :df.drop(column, axis=1, inplace=True) 
    #对于产率过小的行数进行删除处理
   
    while True:
        try:
            df.to_csv(os.path.join(folder_path, 'data_new1.csv'),index=False)
        except PermissionError:
            input('Cannot save file.')
            continue
        else:
            break

else:
    df = pd.read_csv(os.path.join(folder_path, 'data_new_raw.csv'),index=False)
    
print('Hello')

# 分离特征和目标变量

scaler = MinMaxScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
X = df.drop(columns=['product'] )
y = df['product']

# 划分训练集和测试集
for i in range(1):
    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.1)
                    # 初始化并训练随机森林回归模
    model = RandomForestRegressor(n_estimators=200)

    model.fit(X_train, y_train)

                        # 预测测试集
    

    y_pred = model.predict(X_test)
    scorer = make_scorer(r2_score)
    

                        # 评估模型性能

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')
    scores = cross_val_score(model, X, y, cv=5, scoring=scorer)
    print(f"Cross-validation scores: {scores}")
    print(f"Average score: {scores.mean()}")
feature_importances = model.feature_importances_

# 获取特征名称
feature_names = X_train.columns

# 将特征名称和特征重要性组合成一个DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# 按重要性降序排序
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 获取最重要的十个特征
top_ten_features = feature_importance_df.head(10)

# 绘制条形图
plt.figure(figsize=(10, 8))
plt.barh(top_ten_features['Feature'], top_ten_features['Importance'], color='dodgerblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()  # 反转y轴，使得最重要的特征在上方
plt.show()
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Measured vs Predicted')
plt.show()

    # 绘制误差分布图
errors = y_test - y_pred
plt.figure(figsize=(10, 8))
plt.hist(errors, bins=25)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.show()

sys.exit()