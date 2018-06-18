import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras import initializers
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# configuration
epochs = 20
batch_size = 32
val_size = 100
units = 16

dataset = datasets.load_breast_cancer()
X = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

print(x_train[0], x_test[0], y_train[0], y_test[0])

y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

model = None

model = models.Sequential()
model.add(layers.Dense(units, activation='relu', input_shape=(30,), kernel_initializer='random_uniform',
                       bias_initializer=initializers.Constant(value=0.5)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

x_val = x_train[:val_size]
partial_x_train = x_train[val_size:]
y_val = y_train[:val_size]
partial_y_train = y_train[val_size:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val))

model.predict(x_test)
