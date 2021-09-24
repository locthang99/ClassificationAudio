import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# import thư viện
from sklearn.datasets import load_iris

import pandas as pd
import seaborn as sn
from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Thực hiện load dữ liệu
iris_data = load_iris() 

# In ra 10 input đầu tiên
print('First 10 inputs: ')
print(iris_data.data[:10])
# In ra 10 output đầu tiên
print('First 10 output (label): ')
print(iris_data.target[:10])

# Gán input vào biến X
X = iris_data.data
# Gán output vào biến y 
y = iris_data.target.reshape(-1,1)

# Thực hiện Onehot transform
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)
print("Output after transform")
print(y[:10])

# Chia dữ liệu train, test với tỷ lệ 80% cho train và 20% cho test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Khai báo model
model = Sequential()

model.add(Dense(128, input_shape=(4,), activation='relu', name='layer1'))
model.add(Dense(128, activation='relu', name='layer2'))
model.add(Dense(3, activation='softmax', name='output'))

# Cài đặt hàm tối ưu Adam 
optimizer = Adam()
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])


# In cấu trúc mạng ra màn hình
print('Detail of network: ')
print(model.summary())

# Train model
model.fit(X_train, y_train, batch_size=32, epochs=10)

# Kiểm tra trên tập test
results = model.evaluate(X_test, y_test)

print('Test loss: {:4f}'.format(results[0]))
print('Test accuracy: {:4f}'.format(results[1]))

# Train model
import matplotlib.pyplot as pyplot
#history = model.fit(X_train, y_train, batch_size=32, epochs=30,validation_data=(X_test,y_test))

# plot loss during training

# pyplot.figure(figsize=(20,10))
# pyplot.subplot(211)
# pyplot.title('Loss')
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# # plot accuracy during training
# pyplot.subplot(212)
# pyplot.title('Accuracy')
# pyplot.plot(history.history['accuracy'], label='train')
# pyplot.plot(history.history['val_accuracy'], label='test')
# pyplot.legend()
# pyplot.show()

# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix



loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)


print("Acc:")
print(accuracy)
print("f1_score: ")
print(f1_score)
print("precision: ")
print(precision)
print("recal: ")
print(recall)



df_cm = pd.DataFrame(matrix, index = [i for i in "012"],
                  columns = [i for i in "012"])
pyplot.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)