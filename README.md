## 1. �����
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.optimizers import Adam
```
numpy �� pandas �������ݴ���
sklearn.preprocessing �ṩ������Ԥ����Ĺ��ߡ�
keras.models.Sequential �� keras.layers ���ڹ���������ģ�͡�
keras.optimizers.Adam ���Ż��������ڱ���ģ�͡�
## 2. ��ȡ����
```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```
��ȡѵ�����ݺͲ������ݡ�
## 3. ����Ԥ������
```python
def preprocess_data(df, look_back=100):
    grouped = df.groupby('id')
    datasets = {}
    for id, group in grouped:
        datasets[id] = group.values
```
�����ݰ� id ���飬����ÿ�����ݴ洢�� datasets �ֵ��С�
```python
    X, Y = [], []
    for id, data in datasets.items():
        for i in range(10, 15):  # ÿ��id����5������
            a = data[i:(i + look_back), 3]
            a = np.append(a, np.array([0] * (100 - len(a))))
            X.append(a[::-1])
            Y.append(data[i - 10:i, 3][::-1])
```
Ϊÿ�� id ����ѵ�����ݣ��ֱ𹹽�5�����У�������10��14����ÿ�����г���Ϊ100��������100�Ĳ�����0��䣬�������������ӵ� X�����룩�� Y���������
```python
    OOT = []
    for id, data in datasets.items():
        a = data[:100, 3]
        a = np.append(a, np.array([0] * (100 - len(a))))
        OOT.append(a[::-1])

    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64), np.array(OOT, dtype=np.float64)
```
�����������ݼ� OOT��ʹ��ǰ100�����ݣ��������0��
## 4. ����ģ��
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
����LSTMģ�ͣ�����һ��LSTM�㡢һ��RepeatVector�㡢һ���������е�LSTM���һ��TimeDistributed Dense�㡣
ʹ��Adam�Ż����;��������ʧ��������ģ�͡�
## 5. ������ѵ��ģ��
```python
look_back = 100  # ���г���
n_features = 1  # ����ÿ��ʱ���ֻ��һ������
n_output = 10  # Ԥ��δ��10��ʱ�䵥λ��ֵ
```
�������г��ȡ�����������������ȡ�
```python
X, Y, OOT = preprocess_data(train, look_back=look_back)
```
Ԥ�������ݡ�
```python
model = build_model(look_back, n_features, n_output)
```
����ģ�͡�
```python
model.fit(X, Y, epochs=50, batch_size=64, verbose=1)
```
ʹ��ѵ������ѵ��ģ�ͣ�50��epochs��batch sizeΪ64��
## 6. ����Ԥ��
```python
predicted_values = model.predict(OOT)
```
ʹ��ģ�ͶԲ������ݽ���Ԥ�⣬�õ� predicted_values��
## 7. ����չƽԤ��ֵ
```python
print(predicted_values.shape)  # (id��, 10, 1)
```
��ӡԤ��ֵ����״ (id��, 10, 1)��
```python
flattened_values = predicted_values.reshape(predicted_values.shape[0], -1)
```
����ά����չƽ�ɶ�ά���� (id��, 10)��
## 8. ����Ԥ��ֵΪCSV�ļ�
```python
df = pd.DataFrame(flattened_values, columns=[f'Predicted Value {i+1}' for i in range(flattened_values.shape[1])])
df.to_csv('predicted_values.csv', index=False)
```
��չƽ���Ԥ��ֵת��ΪDataFrame��������ΪCSV�ļ���
```python
print("Predicted values have been saved to 'predicted_values.csv'.")
```
��ӡȷ����Ϣ��"# elec_predict" 
