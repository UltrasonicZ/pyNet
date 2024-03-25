from perceptron import Perceptron
import numpy as np
def step(x):
    if x > 0:
        return 1
    return 0

if __name__ == "__main__":
    p = Perceptron(2 ,step)
    inputs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    labels = [1, 0, 0, 0]
    p.train(inputs, labels, 1000, 0.1)
    print("final result:")
    print(p.predict([1, 1]))
    print(p)