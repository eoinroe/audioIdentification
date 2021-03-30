import numpy as np

# matrix = np.arange(16).reshape(4, 4)
#
# sub_array = np.take(matrix, )
#
#
# x = np.arange(20).reshape(5, 4)
# print(x)
#
# row, col = np.indices((3, 3))
#
# print('Row: ', row)
# print('Col: ', col)
#
# print('Output: ', x[row, col])
# # print('Take: ', np.take(np.take(x, row, axis=0), col, axis=1))
# #
# # a = [4, 3, 5, 7, 6, 8]
# # indices = [0, 1, 4]
# #
# # print('Test: ', np.take(a, [[0, 1], [2, 3]]))
#
# matrix = np.arange(9).reshape(3, 3)
# print(matrix)
#
# ixgrid = np.ix_([0, 2], [1, 0])
# print(ixgrid)
#
# print(matrix[ixgrid])
#
# print('Type: ', type(ixgrid))
#
# # print(np.ravel_multi_index(matrix, ixgrid, mode='clip'))

# lut = np.random.randint(11, 99, (13, 13, 13))
# print(lut)
#
# arr = np.arange(12).reshape(2, 2, 3)
# print(arr)
#
# print('Shape: ', lut.shape)
#
# print(np.take(lut, np.ravel_multi_index(arr.transpose(2, 0, 1), lut.shape)))

# matrix = np.arange(9).reshape(3, 3)
# print(matrix)
#
# print(matrix.shape)
#
# row, col = np.indices((3, 3))
# print(np.take(matrix, [[0, 1], [2, 3]], (2, 2)))

# Seems the axis argument of np.take can be the shape of an array:
# https://stackoverflow.com/questions/42573568/numpy-multidimensional-indexing-and-the-function-take

matrix = np.arange(16).reshape(4, 4)
print(matrix)

size = 2

w = matrix.shape[0]
h = matrix.shape[1]

row, col = np.indices((size, size))

for x in range(0, w, size):
    for y in range(0, h, size):
        m = matrix[row + x, col + y]
        print(m)

# print(matrix[row + 2, col + 1])
