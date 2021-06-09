import tensorflow as tf
import numpy as np
import matlab.engine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

class NeuralNetwork:
    checkpoint_path = "training_1/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    def __init__(self, dims):
        self.model = None
        self.weights = []
        self.bias = []
        self.dims = dims
        self.weights_ = None

    def load_data(self):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        # Normalize the images.
        train_images = (train_images / 255) - 0.5
        test_images = (test_images / 255) - 0.5
        # Flatten the images.
        train_images = train_images.reshape((-1, 784))
        test_images = test_images.reshape((-1, 784))
        return train_images, train_labels, test_images, test_labels

    # Creating a Sequential Model and adding the layers
    def create_model(self):
        model = Sequential([
                Dense(self.dims[1], activation='relu', input_shape=(784,))
            ])
        if len(self.dims) >= 4:
            for i in self.dims[2 : -1]:
                model.add(Dense(i, activation= "relu"))
        model.add(Dense(self.dims[-1], activation="softmax"))
        model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        self.model = model

    def train(self, cache):
        train_images, train_labels, test_images, test_labels = self.load_data()
        self.create_model()
        if cache:
            self.model.load_weights(self.checkpoint_path)
        else:
            self.model.fit(x=train_images,y=train_labels, epochs=3, callbacks=[self.cp_callback])
            self.model.load_weights(self.checkpoint_path)

    def read_weights(self):
        W = []
        b = []
        for layer in self.model.layers:
            W.append(matlab.double(layer.get_weights()[0].T.tolist()))
            b.append(matlab.double(np.expand_dims(layer.get_weights()[1], axis=1).tolist()))
            self.weights.append(layer.get_weights()[0].T)
            self.bias.append(np.expand_dims(layer.get_weights()[1], axis =1))
        self.weights_ = W
        self.bias_ = b

    def generateRandomWeights(self):
        for index, dim in enumerate(self.dims[: -1]):
            np.random.seed(index)
            self.weights.append(np.random.rand(self.dims[index + 1], dim))
            self.bias.append(np.random.rand(self.dims[index + 1],1))
        return

    def relu(self, x):
        return np.maximum(x, 0)

    def interval_arithmetic(self, x_min, x_max, method):
        X_min = []
        X_max = []
        Y_min = []
        Y_max = []

        if method == 'SDR':

            X_min.append(x_min)
            X_max.append(x_max)
            for i in range(len(self.dims) - 2):
                Y_min.append((np.matmul(np.maximum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))),
                                        X_min[i]) + np.matmul(
                    np.minimum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))), X_max[i]) + self.bias[
                                  i]))
                Y_max.append((np.matmul(np.maximum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))),
                                        X_max[i]) + np.matmul(
                    np.minimum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))), X_min[i]) + self.bias[
                                  i]))

                X_min.append(self.relu(Y_min[i]))
                X_max.append(self.relu(Y_max[i]))

            return Y_min, Y_max, X_min, X_max

        elif method == 'DeepSDP' or method == 'GlobalLip':
            X_min.append(x_min.T)
            X_max.append(x_max.T)
            for i in range(len(self.dims) - 2):
                Y_min.append((np.matmul(np.maximum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))), X_min[i].T) + np.matmul(np.minimum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))), X_max[i].T) + self.bias[i]).T)
                Y_max.append((np.matmul(np.maximum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))), X_max[i].T) + np.matmul(np.minimum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))), X_min[i].T) + self.bias[i]).T)

                X_min.append(self.relu(Y_min[i]))
                X_max.append(self.relu(Y_max[i]))

            X_min = np.concatenate(X_min[1:], axis = 1)
            X_max = np.concatenate(X_max[1:], axis = 1)
            Y_min = np.concatenate(Y_min[:], axis = 1)
            Y_max = np.concatenate(Y_max[:], axis = 1)
        return Y_min, Y_max, X_min, X_max

