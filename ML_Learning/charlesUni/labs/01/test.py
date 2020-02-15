import numpy as np
a = [[1, 2, 3, 4, 5], [0, 0, 0, 0, 0]]
print(a)  # [1, 2, 3, 4, 5]
b = np.pad(a, (0, 1), 'constant', constant_values=(9))
print(b)  # [0 0 1 2 3 4 5 0 0 0]
