import test
file = ["Neural Network/MNIST/mnist10x20.nnet", "Neural Network/MNIST/mnist20x20.nnet", "Neural Network/MNIST/mnist20x40.nnet"]
# dims = [[784, 20, 10], [784, 50, 10], [784, 50, 50, 10], [784, 100, 100, 10]]
dims = [[10,20,10]]
for item in file:
    test.test(0.1, file_path=item)