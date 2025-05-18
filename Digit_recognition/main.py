import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# from scipy.special import expit as σ

import random
import pickle


SIZE = (784, 40, 32, 16, 10)


def ReLU(X):
  return np.maximum(X, 0)
    

def σ(X):
  return 1.0 / (1.0 + np.exp(-X))


def dReLU(X):
  return X > 0


def dσ(X):
  return σ(X)*(1.0 - σ(X))


def save_model(network: object, output_file="model.pkl"):
  global SIZE
  with open(output_file, "wb") as f:
    pickle.dump({"Size": SIZE, "B": network.B, "W": network.W}, f)


def load_model(network: object, input_file="model.pkl"):
  with open(input_file, "rb") as f:
    data = pickle.load(f)
    network.sizes = data["Size"]
    network.num_layers = len(data["Size"])
    network.B = data["B"]
    network.W = data["W"]



class Network:

  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.B = [np.random.randn(y, 1) for y in sizes[1:]]
    self.W = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


  def y(self, a):
    for b, w in zip(self.B, self.W):
      a = σ((w @ a)+b)
    return a


  def SGD(self, training_data, epochs, mini_batch_size, η, test_data=None):
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
      print(f"Learning rate η: {η}")
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, η)
      # η -= η/(epochs+1)
      if test_data:
        print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
      else:
        print(f"Epoch {j} complete")


  def update_mini_batch(self, mini_batch, η):
    grad_b = [np.zeros(b.shape) for b in self.B]
    grad_w = [np.zeros(w.shape) for w in self.W]
    for x, y in mini_batch:
      Δgrad_b, Δgrad_w = self.backprop(x, y)
      grad_b = [nb+dnb for nb, dnb in zip(grad_b, Δgrad_b)]
      grad_w = [nw+dnw for nw, dnw in zip(grad_w, Δgrad_w)]
    self.W = [w-(η/len(mini_batch))*nw for w, nw in zip(self.W, grad_w)]
    self.B = [b-(η/len(mini_batch))*nb for b, nb in zip(self.B, grad_b)]


  def backprop(self, x, y):
    grad_b = [np.zeros(b.shape) for b in self.B]
    grad_w = [np.zeros(w.shape) for w in self.W]
    # Feedforward
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(self.B, self.W):
      z = w @ activation + b
      zs.append(z)
      activation = σ(z)
      activations.append(activation)

    Δ = self.dC(activations[-1], y) * dσ(zs[-1])
    grad_b[-1] = Δ
    grad_w[-1] = Δ @ activations[-2].T

    for l in range(2, self.num_layers):
      z = zs[-l]
      sp = dσ(z)
      Δ = self.W[-l+1].T @ Δ * sp
      grad_b[-l] = Δ
      grad_w[-l] = Δ @ activations[-l-1].T
    return (grad_b, grad_w)

  def evaluate(self, test_data):
    test_results = [(np.argmax(self.y(x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

  def dC(self, a, y):
    return (a-y)


def load_data_wrapper(training_data, test_data):
  tr_d, te_d = training_data, test_data
  training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
  training_results = [vectorized_result(y) for y in tr_d[1]]
  training_data = list(zip(training_inputs, training_results))
  test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
  test_data = list(zip(test_inputs, te_d[1])) 
  return (training_data, test_data)

def vectorized_result(j):
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e



def train(net, training_data, epochs, batch_size, η, test_data):
  net.SGD(training_data, epochs=epochs, mini_batch_size=batch_size, η=η, test_data=test_data)



training_data_raw = pd.read_csv("./dataset/mnist_train.csv")
training_data_raw = np.array(training_data_raw)

test_data_raw = pd.read_csv("./dataset/mnist_test.csv")
test_data_raw = np.array(test_data_raw)

training_data = (training_data_raw.T[1:].T, training_data_raw.T[0]) # Inputs and desired outputs
# print(training_data)

test_data = (test_data_raw.T[1:].T, test_data_raw.T[0])
# print(training_data[0][0].shape)

# current_image = current_image.reshape((28, 28)) * 255
# plt.gray()
# plt.imshow(current_image, interpolation='nearest')
# plt.show()


training_data, test_data = load_data_wrapper(training_data, test_data)

net = Network(SIZE)

load_model(net, input_file="./model.pkl")

# i = random.randint(0, len(test_data))

# o = net.y(test_data[i][0])
# print(np.argmax(o))

# current_image = test_data[i][0]
# current_image = current_image.reshape((28, 28)) * 255
# plt.gray()
# plt.imshow(current_image, interpolation='nearest')
# plt.show()


train(net, training_data, epochs=30, batch_size=200, η=0.001, test_data=test_data)

# save_model(net)

# η = 0.014 - 0.016
# η = 0.018 - 0.023
# η = 0.027 - 0.041 - 0.063


######################
# First model: 
# η = 0.063 mini_batch_size=10, epochs: 30 - 30 - 15; 
# η = 0.041 mini_batch_size=10, epochs: 30 - 30 - 30;
# η = 0.041 mini_batch_size=100, epochs: 15 - 15 - 60 - 60;
