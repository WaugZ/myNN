import numpy as np

a = np.array(([1, 0, -1], [2, 3, 0]))
b = np.zeros(a.shape)
b[a > 0] = a[a > 0]
c = np.array(a)
print c
c[0, 0] = -2
print a * c
print a
print b, b.shape[0]