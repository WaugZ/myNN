import numpy as np
from scipy import sparse
from itertools import islice


class myNN:

    def __init__(self, num_feature, alpha):
        self.lr = alpha
        self.m = num_feature
        self.W = [0]  # not use W[0]
        self.b = [0]  # not use b[0]
        ''' Make as many layers as you like '''
        w = np.random.rand(3, self.m) * .01
        self.W.append(w)
        b = np.zeros((3, 1))
        self.b.append(b)


    def sigmoid(self, Z):
        # print(Z[0:5, :])
        return 1 / (1 + np.exp(- Z + 1e-10))

    def reLU(self, Z):
        # print Z.shape
        A = np.mat(Z)
        A[A <= 0] = 0
        return A

    def train(self, train_data, labels, max_iter):
        for i in range(max_iter):
            ''' Forward prop '''
            # print self.W1.shape, train_data.shape
            Z = [0]  #  not use Z[0]
            A = [train_data]
            for ind in range(len(self.W) - 2):  # the active function of last  layer is different
                z = self.W[ind + 1] * A[ind] + self.b[ind + 1]
                a = self.reLU(z)
                A.append(a)
                Z.append(z)

            z_last = self.W[-1:] * A[-1:] + self.b[-1:]
            a_last = self.sigmoid(z_last)
            Z.append(z_last)
            A.append(a_last)

            '''cost'''
            J = - np.sum(np.multiply(labels, np.log(a_last + 1e-10)) + np.multiply((1 - labels), np.log(1 - a_last + 1e-10))) / self.m
            print('Iteration ' + str(i) + ' cost = ' + str(J))

            ''' Backward prop '''
            dZ = []
            dW = []
            dB = []
            g = []
            for ind in range(len(A))[::-1] :

            dZ4 = labels - A4
            dW4 = dZ4 * A3.T / m
            db4 = np.sum(dZ4, axis=1, keepdims=True) / self.m
            g3 = np.zeros(A3.shape)
            g3[A3 != 0] = 1
            dZ3 = np.multiply(self.W4.T * dZ4, g3)
            dW3 = dZ3 * A2.T / m
            db3 = np.sum(dZ3, axis=1, keepdims=True) / self.m
            g2 = np.zeros(A2.shape)
            g2[A2 != 0] = 1
            dZ2 = np.multiply(self.W3.T * dZ3, g2)
            dW2 = dZ2 * A1.T / m
            db2 = np.sum(dZ2, axis=1, keepdims=True) / self.m
            g1 = np.zeros(A1.shape)
            g1[A1 != 0] = 1
            dZ1 = np.multiply(self.W2.T * dZ2, g1)
            dW1 = dZ1 * train_data.T / m                    # train_data is A0
            db1 = np.sum(dZ1, axis=1, keepdims=True) / self.m

            ''' Gradient descent '''
            self.W1 = self.W1 + self.lr * dW1
            self.b1 = self.b1 + self.lr * db1
            self.W2 = self.W2 + self.lr * dW2
            self.b2 = self.b2 + self.lr * db2
            self.W3 = self.W3 + self.lr * dW3
            self.b3 = self.b3 + self.lr * db3
            self.W4 = self.W4 + self.lr * dW4
            self.b4 = self.b4 + self.lr * db4

    def coss_val(self, cv_data, labels):
        Z1 = self.W1 * cv_data + self.b1
        A1 = self.reLU(Z1)
        Z2 = self.W2 * A1 + self.b2
        A2 = self.reLU(Z2)
        Z3 = self.W3 * A2 + self.b3
        A3 = self.reLU(Z3)
        Z4 = self.W4 * A3 + self.b4
        A4 = self.sigmoid(Z4)
        J = np.sum(labels * np.log(A4) + (1 - labels) * np.log(1 - A4)) / self.m
        return J

    def pridect (self, test_data):
        Z1 = self.W1 * test_data + self.b1
        A1 = self.reLU(Z1)
        Z2 = self.W2 * A1 + self.b2
        A2 = self.reLU(Z2)
        Z3 = self.W3 * A2 + self.b3
        A3 = self.reLU(Z3)
        Z4 = self.W4 * A3 + self.b4
        A4 = self.sigmoid(Z4)
        return A4


if __name__ == '__main__':
    print('loading training data...')
    file = open('/Users/wangzi/PycharmProjects/test/large_scale/train_data.txt')
    train_X = []
    train_y = []
    train_row_index = []
    train_col_index = []
    count = 0

    for line in islice(file, 0, 1e4):
        data_line = str(line).split()
        train_y.append(int(data_line[0]))
        for i in range(1, len(data_line)):
            index_value = data_line[i].split(':')
            index = int(index_value[0])
            value = float(index_value[1])
            # it is said that test example does not contain more than 132 features -- mahua
            if index > 132:
                break
            train_X.append(value)
            train_row_index.append(count)
            train_col_index.append(index)
        count = count + 1

    file.close()

    data_X = sparse.csr_matrix((train_X, (train_row_index, train_col_index))).todense().T
    data_y = np.array(train_y)
    data_y = data_y.reshape(data_y.shape[0], 1).T

    print data_X.shape, data_y.shape
    m, n = data_X.shape
    n = int(.8 * n)
    train_X = data_X[:, :n]
    train_y = data_y[:, :n]
    cv_X = data_X[:, n:]
    cv_y = data_y[:, n:]
    clf = myNN(train_X.shape[0], 0.001)
    clf.train(train_X, train_y, 1000)
    p = clf.pridect(cv_X)
    p[p > .5] = 1
    p[p <= .5] = 0
    print n, np.sum(p != cv_y)

