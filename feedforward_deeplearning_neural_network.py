import numpy as np
import random as Ran

class Neural_Network:
    def __init__(self, layersizes):
        weight_shapes = [(a,b) for a,b in zip(layersizes[1:], layersizes[:-1])]
        self.weights = [np.random.standard_normal(s)/s[1]**.5 for s in weight_shapes]
        self.biases = [np.zeros((s,1)) for s in layersizes[1:]]
        self.layersizes = layersizes
    
    def feedforward(self, I):
        for w,b in zip(self.weights, self.biases):
            I = self.activation(np.matmul(w, I) + b)
        return I
    
    def calculate_accuracy(self, data):
        accuracy = 0.0
        for sample in data:
            self_output = self.feedforward(sample[0])
            if np.argmax(self_output) == np.argmax(sample[1]):
                accuracy += 1
        accuracy = accuracy / len(data) * 100
        return accuracy

    def backprop(self, input, output):
        
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        Activation = input
        Activations = [input]
        Z_value = 0.0
        Z_values = []

        for b, w in zip(self.biases, self.weights):
            Z_value = np.matmul(w, Activation) + b
            Activation = self.activation(Z_value)

            Activations.append(Activation)
            Z_values.append(Z_value)

        Activation_derivative = self.activation_prime(Z_values[-1])
        Cost_output_delta = (Activations[-1] - output)
        delta = Cost_output_delta * Activation_derivative
        transpose_value = np.transpose(self.weights[-2])

        gradient_b[-1] = delta
        gradient_w[-1] = np.matmul(delta, np.transpose(Activations[-2]))

        for i in range(2, len(self.layersizes)):

            Z_value = Z_values[-i]
            Activation_derivative = self.activation_prime(Z_value)
            transpose_value = np.transpose(self.weights[-i+1])

            delta = [
                (a * b) for a,b in zip(np.dot(transpose_value, delta), Activation_derivative)
            ]

            gradient_b[-i] = delta
            gradient_w[-i] = np.matmul(delta, np.transpose(Activations[-i-1]))
        
        return (gradient_b, gradient_w)


    def stochastic_gradient_descent(self, Training_data, Epochs, mini_batch_size, eta):

        for i in range(Epochs):
            Ran.shuffle(Training_data)
            mini_batches = [
                Training_data[k:k+mini_batch_size]
                for k in range(0, len(Training_data))
            ]

            for mini_batch in mini_batches:
                self.Update_mini_batch(mini_batch, eta)

            print("Epoch {0} complete".format(i+1))

    def Update_mini_batch(self, mini_batch, eta):

        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        for input, output in mini_batch:
            delta_gradient_pair = self.backprop(input, output)
            delta_gradient_b = delta_gradient_pair[0]
            delta_gradient_w = delta_gradient_pair[1]

            Bias_zip = zip(gradient_b, delta_gradient_b)
            Weight_zip = zip(gradient_w, delta_gradient_w)

            gradient_b = [g_b + d_b for g_b, d_b in Bias_zip]
            gradient_w = [g_w + d_w for g_w, d_w in Weight_zip]
        

        Bias_zip = zip(self.biases, gradient_b)
        Weight_zip = zip(self.weights, gradient_w)

        self.biases = [b - (eta / len(mini_batch) * g_b) for b, g_b in Bias_zip]
        self.weights = [w - (eta / len(mini_batch) * g_w) for w, g_w in Weight_zip]

    
    def activation(self, value):
        return 1 / (1 + np.exp(-value))
    
    def activation_prime(self, value):
        return np.exp(-value) / ((1 + np.exp(-value))**2)
