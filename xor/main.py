# XOR
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras import Model


class XorModel(Model):

    layerList = []

    def __init__(self):
        super(XorModel, self).__init__()

        # layers
        self.dense = Dense(16, input_dim=2, activation='relu')
        self.dense2 = Dense(16, activation='relu')
        self.sigmoid = Dense(1, activation='sigmoid')

        self.layerList.append(self.dense)
        self.layerList.append(self.dense2)
        self.layerList.append(self.sigmoid)

    def call(self, x):
        for layer in self.layerList:
            x = layer(x)

        return x


print("Tensorflow version: " + tf.__version__)

training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [1, 0], [1, 0], [1, 1], [0, 0]], "float32")
training_target_data = np.array([[0], [1], [1], [0], [1], [1], [1], [0], [0]], "float32")

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
test_target_data = np.array([[0], [1], [1], [0]], "float32")

model = XorModel()
model.compile(loss='mean_squared_error', #tf.keras.metrics.Mean(name='train_loss')
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['binary_accuracy'])
print("Model fit")
model.fit(training_data, training_target_data, nb_epoch=500, verbose=1)
print("Model evaluate")
model.evaluate(test_data, test_target_data)

print("Prediction with test data")
print(model.predict(np.array([[1, 0], [0, 1], [0, 0], [1, 1]], "float32")).round())

print("\n\nModell Summary")
model.summary()
print("\n\n")
