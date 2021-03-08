import numpy as np

a = np.zeros((3, 3), dtype=bool)
print(a)
a[1:3, 1:3] = True
print(a)