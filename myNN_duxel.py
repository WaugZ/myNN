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
        w = np.random.rand(1, self.m) * .01
        self.W.append(w)
        b = np.zeros((1, 1))
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
            # print self.W[1].shape, train_data.shape
            Z = [0]  #  not use Z[0]
            A = [train_data]
            for ind in range(len(self.W) - 1):  # the active function of last  layer is different
                z = np.dot(self.W[ind + 1], A[ind]) + self.b[ind + 1]
                a = self.reLU(z)
                A.append(a)
                Z.append(z)

            # print self.W[1].shape, A[0].shape
            z_last = np.dot(self.W[len(self.W) - 1], A[len(A) - 2]) + self.b[len(self.b) - 1]
            # print z_last.shape
            a_last = self.sigmoid(z_last)
            Z.append(z_last)
            A.append(a_last)

            '''cost'''
            J = - np.sum(np.multiply(labels, np.log(a_last + 1e-10)) + np.multiply((1 - labels), np.log(1 - a_last + 1e-10))) / self.m
            print('Iteration ' + str(i) + ' cost = ' + str(J))

            ''' Backward prop '''
            dZ = [labels - a_last] * len(Z)
            dW = [0] * len(self.W)
            dB = [0] * len(self.b)
            g = [0] * len(Z)
            for ind in range(len(A) - 1)[::-1]:
                # print ind
                if ind < 1:
                    break
                # print dZ[ind].shape
                dW[ind] = np.dot(dZ[ind], A[ind - 1].T) / self.m
                dB[ind] = np.sum(dZ[ind], axis=1) / self.m
                # print dB[ind].shape
                g[ind - 1] = np.zeros(A[ind - 1].shape)
                g[ind - 1][A[ind - 1] != 0] = 1
                dZ[ind - 1] = np.multiply(self.W[ind].T * dZ[ind], g[ind - 1])


            ''' Gradient descent '''
            for ind in range(1, len(self.W)):
                self.W[ind] += self.lr * dW[ind]
                self.b[ind] += self.lr * dB[ind]

    def coss_val(self, cv_data, labels):
        Z = [0]
        A = [cv_X]
        for ind in range(1, len(self.W) - 1):
            z = np.dot(self.W[ind], A[ind - 1]) + self.b[ind]
            a = self.reLU(z)
            Z.append(z)
            A.append(a)
        # print len(self.W)
        z_last = self.W[len(self.W) - 1] * A[len(A) - 2] + self.b[len(self.b) - 1]
        a_last = self.sigmoid(z_last)

        J = - np.sum(np.multiply(labels, np.log(a_last + 1e-10)) + np.multiply((1 - labels),
                                                                               np.log(1 - a_last + 1e-10))) / self.m
        return J

    def pridect (self, test_data):
        Z = [0]
        A = [test_data]
        for ind in range(1, len(self.W) - 1):
            z = np.dot(self.W[ind], A[ind - 1]) + self.b[ind]
            a = self.reLU(z)
            Z.append(z)
            A.append(a)
        # print len(self.W)
        z_last = self.W[len(self.W) - 1] * A[len(A) - 2] + self.b[len(self.b) - 1]
        a_last = self.sigmoid(z_last)
        return a_last


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

    # print data_X.shape, data_y.shape
    m, n = data_X.shape
    n = int(.8 * n)
    train_X = data_X[:, :n]
    train_y = data_y[:, :n]
    cv_X = data_X[:, n:]
    cv_y = data_y[:, n:]
    clf = myNN(train_X.shape[0], 0.0001)
    clf.train(train_X, train_y, 1000)
    p = clf.pridect(cv_X)
    p[p > .5] = 1
    p[p <= .5] = 0
    print int(.2 * n), np.sum(p != cv_y)

