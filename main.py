import numpy as np
import librosa.display
from skimage.feature import peak_local_max
from matplotlib import pyplot as plt
from scipy import ndimage
import math
import itertools


def peak_picking_max_filter(signal, min_distance, threshold_rel):
    """
        Detect peaks from an STFT spectrogram.  The minimum intensity is calculated in the same
        way as skimage peak_local_max, i.e. as max(image) * threshold_rel.  Also, the docs state
        'Peaks are the local maxima in a region of 2 * min_distance + 1', so the value of the
        size parameter for ndimage.maximum_filter is calculated in the same way.

                Parameters:
                    signal (np.ndarray):
                    min_distance (int):
                    threshold_rel (float):
    """
    matrix_d = np.abs(librosa.stft(signal, n_fft=1024, window='hann', win_length=1024, hop_length=512))

    min_intensity = threshold_rel * np.amax(D)
    size = min_distance * 2 + 1

    # Apply maximum filter to the matrix using a 2D neighborhood
    arr = ndimage.maximum_filter(matrix_d, size=size)

    # Assigning nan to any value lower than the minimum intensity
    arr[arr < min_intensity] = math.nan

    # Return a logical array
    return matrix_d == arr


def peak_picking_fpm(signal, min_distance, threshold_rel):
    """
        Peak picking based on the method describe in Fundamentals of
        Music Processing.

                Parameters:
                    signal (np.ndarray):
                    min_distance (int):
                    threshold_rel (float):
    """
    matrix_d = np.abs(librosa.stft(signal, n_fft=1024, window='hann', win_length=1024, hop_length=512))

    min_intensity = threshold_rel * np.amax(matrix_d)
    size = min_distance

    # Convert tuple to ndarray
    shape = np.asarray(matrix_d.shape)
    remainder = shape % size

    # If one of the dimensions is not divisible by the neighborhood size
    # add padding to the edge of the the corresponding axis.
    pad_width = np.where(remainder > 0, size - remainder, 0).astype(int)

    grid = np.pad(matrix_d, ((0, pad_width[0]), (0, pad_width[1])), constant_values=(0, 0))

    w = grid.shape[0] // size
    h = grid.shape[1] // size

    for i, j in itertools.product(range(w), range(h)):
        x = i * size
        y = j * size

        # Create a sub array for the current neighborhood
        m = grid[x:x + size, y:y + size]

        # Find the maximum value in the neighborhood
        maximum = np.amax(m)

        # Assigning nan to any value lower than the minimum intensity
        m[m < min_intensity] = math.nan

        # Assign truth values to all the values in the neighborhood
        grid[x:x + size, y:y + size] = m == maximum

    # Rows and columns to delete from the matrix to transform back to original shape
    rows = slice(grid.shape[0] - pad_width[0], grid.shape[0])
    cols = slice(grid.shape[1] - pad_width[1], grid.shape[1])

    grid = np.delete(grid, np.s_[rows], axis=0)
    grid = np.delete(grid, np.s_[cols], axis=1)

    # Could potentially just use the indices of the original matrix to obtain
    # the truth values

    return grid


# Investigate different peak picking options for creating constellation maps,
# and use the two lab recordings to plot the derived constellation maps.

# Load the query recording
snd, sr = librosa.load('data/jazz.00005-snippet-10-0.wav')

# Detect peaks from STFT spectrogram and plot constellation map
D = np.abs(librosa.stft(snd, n_fft=1024, window='hann', win_length=1024, hop_length=512))
coordinates = peak_local_max(np.log(D), min_distance=10, threshold_rel=0.05, indices=False)

plt.figure(figsize=(10, 5))
plt.imshow(coordinates, cmap=plt.get_cmap('gray_r'), origin='lower')
plt.show()

# coordinates = peak_picking_max_filter(y, min_distance=10, threshold_rel=0.05)
coordinates = peak_picking_fpm(y, min_distance=20, threshold_rel=0.05)

plt.figure(figsize=(10, 5))
plt.imshow(coordinates, cmap=plt.get_cmap('gray_r'), origin='lower')
plt.show()

# Experiment with different time-frequency representations beyond the STFT,
# e.g. you can use the CQT spectrogram or mel spectrogram as alternatives.
