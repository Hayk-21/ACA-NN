from tkinter import N
import numpy as np

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        param input_size: integer, number of features of the input
        param hidden_size: integer, arbitrary number of parameters
        param output_size: integer, number of classes

        Define simple two layer neural network with relu activation function.

        You need to create weights and biases for both layers with the correct 
        shapes. Pass values to self.params dict for later use.
        '''
        self.params = {}
        '''
        START YOUR CODE HERE
        '''
        self.params['W1'] = np.random.randn(hidden_size, input_size)
        self.params['W2'] = np.random.randn(output_size, hidden_size)
        self.params['b1'] = np.random.randn(hidden_size)
        self.params['b2'] = np.random.randn(output_size)
        '''
        END YOUR CODE HERE
        '''


    def loss(self, X, y, reg=0.0):
        '''
        param X: numpy.array, input features
        param y: numpy.array, input labels
        param reg: float, regularization value


        Return:
        param loss: Define loss with data loss and regularization loss
        param grads: Gradients for weights and biases
        '''

        # Unpack weights and biases
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        loss = None
        grads = {}

        '''
        START YOUR CODE HERE
        '''
        h = X.dot(W1) + b1
        h_relu = np.maximum(h, 0)
        y_pred = self.predict(X)
        loss = np.square(y_pred - y).sum() + reg*(W1.dot(W1.T) + W2.dot(W2.T))
        
        '''
        END YOUR CODE HERE
        '''
        grads['y_pred'] = 2.0*(y_pred - y)
        grads['W2'] = h_relu.T.dot(grads['y_pred'])
        grads['h_relu'] = grads['y_pred'].dot(W2.T)
        grads['h'] = grads['h_relu'].copy()
        grads['h'][h < 0] = 0
        grads['W1'] = X.T.dot(grads['h'])
        grads['b1'] = grads['h']
        grads['b2'] = grads['y_pred']
        
        return loss, grads


    def train(self, X_train, y_train, X_val, y_val, learning_rate=1e-3, batch_size=4, num_iters=100):
        '''
        param X_train: numpy.array, trainset features 
        param y_train: numpy.array, trainset labels
        param X_val: numpy.array, valset features
        param y_val: numpy.array, valset labels
        param learning_rate: float, learning rate should be used to updated grads
        param batch_size: float, batch size is the number of images should be used in single iteration
        param num_iters: int, number of iterations you want to train your model

        method will return results and history of the model.
        '''

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in num_iters:
            # Create batches
            X_batch, y_batch = None, None
            '''
            START YOUR CODE HERE
            '''
            
            '''
            END YOUR CODE HERE
            '''
            
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # update weights and biases
            '''
            START YOUR CODE HERE
            '''
            grad['W1'] -= learning_rate * grad['W1']
            grad['W2'] -= learning_rate * grad['W2']
            grad['b1'] -= learning_rate * grad['b1']
            grad['b2'] -= learning_rate * grad['b2']
            '''
            END YOUR CODE HERE
            '''
            if (it+1) % 100 == 0:
                print(f'Iteration {it+1} / {num_iters} : {loss}')
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
            

        return {'loss_history': loss_history, 'train_acc_history': train_acc_history, 'val_acc_history': val_acc_history}

    def predict(self, X):
        '''
        param X: numpy.array, input features matrix
        return y_pred: Predicted values

        Use trainied weights to do prediction for the given features 
        '''
        y_pred = None

        '''
        START YOUR CODE HERE
        '''
        W1, b1 = self.params['W1'].T, self.params['b1']
        W2, b2 = self.params['W2'].T, self.params['b2']

        z1 = X.dot(W1) + b1
        z1_relu = np.maximum(z1, 0)
        z2 = z1_relu.dot(W2) + b2
        # y_pred = np.argmax(z2)
        y_pred = z2
        print(y_pred.shape)
        '''
        END YOUR CODE HERE
        '''
        return y_pred


