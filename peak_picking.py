import numpy as np
import itertools


def peak_picking(matrix, size):
    # If one of the dimensions is not divisible by the neighborhood
    # size add padding to the edges of the grid.
    remainder = np.asarray(matrix.shape) % size
    pad_width = np.where(remainder > 0, size - remainder, 0).astype(int)

    # Add padding to the right and bottom of the grid if necessary
    grid = np.pad(matrix, ((0, pad_width[0]), (0, pad_width[1])), constant_values=(0, 0))
    print(grid)

    w, h = grid.shape
    row, col = np.indices((size, size))

    for x, y in itertools.product(range(0, w, size), range(0, h, size)):
        m = grid[row + x, col + y]
        print(m)

        # Find the maximum value in the neighborhood
        maximum = np.amax(m)

        # Assigning nan to any value lower than the minimum intensity
        # m[m < min_intensity] = math.nan

        # Assign truth values to all the values in the neighborhood
        grid[row + x, col + y] = m == maximum

    rows_to_delete = slice(grid.shape[0] - pad_width[0], grid.shape[0])
    cols_to_delete = slice(grid.shape[1] - pad_width[1], grid.shape[1])

    grid = np.delete(grid, rows_to_delete, axis=0)
    grid = np.delete(grid, cols_to_delete, axis=1)

    return grid


arr = np.arange(16).reshape(4, 4)
print(peak_picking(arr, 3))
