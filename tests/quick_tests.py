import numpy as np

arr = np.array([1, 2, 3])

print(np.argwhere(arr[:-1] != arr[1:]).reshape(-1))
