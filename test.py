import numpy as np

a = np.array(([1, 0, -1], [2, 3, 0]))
b = np.zeros(a.shape)
b[a > 0] = a[a > 0]
c = np.array(a)
d = [1] * 5
d.append(a)
print a, b, np.dot(a.T, b)