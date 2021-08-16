import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def trinity(x, l, u):
    return np.multiply(u, (x - l)) / (u - l)

def lower_trinity(x, l, u):
    return np.multiply(u, x) / (u - l)
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,9))
# # fig.set_figheight(5)
# # fig.set_figwidth(20)
#
# X = np.arange(-10, 10, 0.1)
# y = np.tanh(X)
# ax[0].plot(X, y, c="orange")
# ax[0].grid()
X = np.arange(-10, 10, 0.1)
y = relu(X)
# ax[1].plot(X, y, c="orange")
# ax[1].grid()
#
# y = sigmoid(X)
# ax[2].plot(X, y, c="orange")
# ax[2].grid()


plt.plot(X, y, c="orange")

y = trinity(X, -10, 10)
plt.plot(X, y)

y = lower_trinity(X, -10, 10)
plt.plot(X, y)
plt.grid()
plt.show()