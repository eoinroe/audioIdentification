import numpy as np
import itertools

# import librosa.display
# from skimage.feature import peak_local_max
# from matplotlib import pyplot as plt
# from scipy import ndimage

grid = np.random.randint(100, size=(5, 4))
print(grid)

size = 4

# Convert tuple to ndarray
shape = np.asarray(grid.shape)
remainder = shape % size

# If one of the dimensions is not divisible by the neighborhood size
# add padding to the edge of the the corresponding axis.
pad_width = np.where(remainder > 0, size - remainder, 0).astype(int)

print('Pad width: ', pad_width)

grid = np.pad(grid, ((0, pad_width[0]), (0, pad_width[1])), constant_values=(0, 0))
print(grid)

shape = np.asarray(grid.shape)
print(shape)

w = grid.shape[0] // size
h = grid.shape[1] // size

# w = grid.shape[0] // size
# h = grid.shape[1] // size

# for i in range(w):
#     for j in range(h):
#         x = i * size
#         y = j * size
#         m = grid[x:x + size, y:y + size]
#         print(m)

# for i, j in zip(range(w), range(h)):
#     x = i * size
#     y = j * size
#     m = grid[x:x + size, y:y + size]
#     print(m)

for i, j in itertools.product(range(w), range(h)):
    x = i * size
    y = j * size
    m = grid[x:x + size, y:y + size]
    print(m)
    grid[x:x + size, y:y + size] = grid[x:x + size, y:y + size] == np.amax(m)

print(grid)

# grid = np.delete(grid, slice(grid.shape[0] - pad_width[0], grid.shape[0]))
# grid = np.delete(grid, slice(grid.shape[1] - pad_width[1], grid.shape[1]))

print(grid.shape[0] - pad_width[0])

# Rows and columns to delete from the matrix to transform back to original shape
rows = slice(grid.shape[0] - pad_width[0], grid.shape[0])
cols = slice(grid.shape[1] - pad_width[1], grid.shape[1])

print(rows)
print(cols)

grid = np.delete(grid, np.s_[rows], axis=0)
grid = np.delete(grid, np.s_[cols], axis=1)

print(grid)
