import os
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# 指定包含 JSON 文件的本地文件夹路径
folder_path = 'C:\\Users\\35078\\OneDrive\\文档\\GitHub\\ordjson'

# 初始化一个空的列表来存储数据
data_list = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {file_path}")  # 调试信息

        # 读取 JSON 文件
        with open(file_path, 'r') as file:
            data = json.load(file)

            # 提取 SMILES、reagent、catalyst 和 yield 信息
            smiles_list = []
            reagents = []
            catalysts = []
            yield_value = None

            for input_type, input_data in data['inputs'].items():
                for component in input_data['components']:
                    for identifier in component['identifiers']:
                        if identifier['type'] == 'SMILES':
                            smiles_list.append(identifier['value'])
                    if input_type == 'Base':
                        reagents.append(component['amount']['moles']['value'])
                    elif input_type == 'metal and ligand':
                        catalysts.append(component['amount']['moles']['value'])

            for outcome in data.get('outcomes', []):
                for product in outcome.get('products', []):
                    if product.get('is_desired_product', False):
                        for measurement in product.get('measurements', []):
                            if measurement['type'] == 'YIELD':
                                yield_value = measurement['percentage']['value']

            # 计算分子指纹和描述符
            for smiles in smiles_list:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule:
                    fingerprint = GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
                    fingerprint_bits = list(fingerprint)

                    # 计算分子描述符
                    descriptors = {desc_name: desc_func(molecule) for desc_name, desc_func in Descriptors.descList}

                    # 将数据添加到列表中
                    data_list.append({
                        'fingerprint': fingerprint_bits,
                        'reagent': sum(reagents),
                        'catalyst': sum(catalysts),
                        'yield': yield_value,
                        **descriptors
                    })
                else:
                    print(f"Invalid SMILES: {smiles}")

# 检查数据列表是否为空
if not data_list:
    print("No data loaded. Please check the JSON files and folder path.")
else:
    print(f"Loaded {len(data_list)} records.")

# 将数据转换为 pandas 数据框
df = pd.DataFrame(data_list)

# 将分子指纹展开为单独的列
fingerprint_df = pd.DataFrame(df['fingerprint'].tolist())
df = df.drop(columns=['fingerprint']).join(fingerprint_df)

# 删除包含 NaN 值的列
df.dropna(axis=1, inplace=True)

# 将所有列名转换为字符串类型
df.columns = df.columns.astype(str)

# 分离特征和目标变量
X = df.drop(columns=['yield'])
y = df['yield']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义参数网格
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}

# 使用网格搜索进行超参数调优
grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"Best parameters: {grid_search.best_params_}")

# 使用最佳参数训练模型
best_svr = grid_search.best_estimator_
best_svr.fit(X_train, y_train)

# 预测测试集
y_pred = best_svr.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')