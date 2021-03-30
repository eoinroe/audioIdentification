import numpy as np

m = np.arange(9).reshape(3, 3)

# m = np.pad(m, ((0, 1), (0, 1)))

# m = np.pad(m, (0, 1))

# m = np.pad(m, ((1,), (2,)))

size = 4

remainder = np.asarray(m.shape) % size
pad_width = np.where(remainder > 0, size - remainder, 0)

# pad_width = np.split(pad_width, [0, 1])

# pad_width = tuple(map(tuple, pad_width))
print(pad_width)

m = np.pad(m, pad_width)
# m = np.pad(m, ((0, 1), (0, 1)))

# m = np.pad(m, [(0, 1), (0, 2)])


zeros = np.array([0, 0])
arr = np.array([1, 1])



print(m)
