# https://mp.weixin.qq.com/s/D7B3ca3eVrelMjEjhXFlSg

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('Dataset.csv')

# 划分特征和目标变量
X = df.drop(['target'], axis=1)
y = df['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,crandom_state=42, stratify=df['target'])

df.head()