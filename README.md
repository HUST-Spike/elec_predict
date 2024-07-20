## 1. 导入库
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.optimizers import Adam
```
numpy 和 pandas 用于数据处理。
sklearn.preprocessing 提供了数据预处理的工具。
keras.models.Sequential 和 keras.layers 用于构建神经网络模型。
keras.optimizers.Adam 是优化器，用于编译模型。
## 2. 读取数据
```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```
读取训练数据和测试数据。
## 3. 数据预处理函数
```python
def preprocess_data(df, look_back=100):
    grouped = df.groupby('id')
    datasets = {}
    for id, group in grouped:
        datasets[id] = group.values
```
将数据按 id 分组，并将每组数据存储在 datasets 字典中。
```python
    X, Y = [], []
    for id, data in datasets.items():
        for i in range(10, 15):  # 每个id构建5个序列
            a = data[i:(i + look_back), 3]
            a = np.append(a, np.array([0] * (100 - len(a))))
            X.append(a[::-1])
            Y.append(data[i - 10:i, 3][::-1])
```
为每个 id 构建训练数据，分别构建5个序列（从索引10到14），每个序列长度为100。将不足100的部分用0填充，并将其逆序后添加到 X（输入）和 Y（输出）。
```python
    OOT = []
    for id, data in datasets.items():
        a = data[:100, 3]
        a = np.append(a, np.array([0] * (100 - len(a))))
        OOT.append(a[::-1])

    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64), np.array(OOT, dtype=np.float64)
```
构建测试数据集 OOT，使用前100个数据，逆序并填充0。
## 4. 定义模型
```python
def build_model(look_back, n_features, n_output):
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, n_features)))
    model.add(RepeatVector(n_output))
    model.add(LSTM(50, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mean_squared_error', optimizer=Adam(0.001))
    return model
```
构建LSTM模型，包含一个LSTM层、一个RepeatVector层、一个返回序列的LSTM层和一个TimeDistributed Dense层。
使用Adam优化器和均方误差损失函数编译模型。
## 5. 构建和训练模型
```python
look_back = 100  # 序列长度
n_features = 1  # 假设每个时间点只有一个特征
n_output = 10  # 预测未来10个时间单位的值
```
定义序列长度、特征数量和输出长度。
```python
X, Y, OOT = preprocess_data(train, look_back=look_back)
```
预处理数据。
```python
model = build_model(look_back, n_features, n_output)
```
构建模型。
```python
model.fit(X, Y, epochs=50, batch_size=64, verbose=1)
```
使用训练数据训练模型，50个epochs，batch size为64。
## 6. 进行预测
```python
predicted_values = model.predict(OOT)
```
使用模型对测试数据进行预测，得到 predicted_values。
## 7. 检查和展平预测值
```python
print(predicted_values.shape)  # (id数, 10, 1)
```
打印预测值的形状 (id数, 10, 1)。
```python
flattened_values = predicted_values.reshape(predicted_values.shape[0], -1)
```
将三维数组展平成二维数组 (id数, 10)。
## 8. 保存预测值为CSV文件
```python
df = pd.DataFrame(flattened_values, columns=[f'Predicted Value {i+1}' for i in range(flattened_values.shape[1])])
df.to_csv('predicted_values.csv', index=False)
```
将展平后的预测值转换为DataFrame，并保存为CSV文件。
```python
print("Predicted values have been saved to 'predicted_values.csv'.")
```
打印确认信息。"# elec_predict" 
