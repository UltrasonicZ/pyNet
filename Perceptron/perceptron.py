import numpy as np
class Perceptron(object):
    def __init__(self, input_nums, func) -> None:
        self.activator = func
        self.weights = np.zeros(input_nums)
        self.bias = 0.0
    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights: {0}   ' \
               'bias: {1}\n'.format(self.weights, self.bias)
    
    def predict(self, input_vec):
        # zipped = list(zip(input_vec, self.weights))
        
        # sum_total = sum(list(map(lambda m : m[0] * m[1], zipped)))
        print(input_vec)
        print(self.weights)
        sum_total = sum(self.weights * input_vec)
        print(sum_total)
        return self.activator(sum_total + self.bias)
    
    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)    

    def _one_iteration(self, input_vecs, labels, rate):
        samples = list(zip(input_vecs, labels))
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            delta = label - output
            self.weights += rate * delta * input_vec
            self.bias += rate * delta 