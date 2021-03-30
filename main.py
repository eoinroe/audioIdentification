import numpy as np
import librosa.display
from skimage.feature import peak_local_max
from matplotlib import pyplot as plt
from scipy import ndimage
import math
import itertools


def peak_picking_max_filter(matrix, min_distance, threshold_rel):
    """
        Detect peaks from an STFT spectrogram.  The minimum intensity is calculated in the same
        way as skimage peak_local_max, i.e. as max(image) * threshold_rel.  Also, the docs state
        'Peaks are the local maxima in a region of 2 * min_distance + 1', so the value of the
        size parameter for ndimage.maximum_filter is calculated in the same way.

                Parameters:
                    matrix (np.ndarray):
                    min_distance (int):
                    threshold_rel (float):
    """
    min_intensity = threshold_rel * np.amax(matrix)
    size = min_distance * 2 + 1

    # Apply maximum filter to the matrix using a 2D neighborhood
    arr = ndimage.maximum_filter(matrix, size=size)

    # Assigning nan to any value lower than the minimum intensity
    arr[arr < min_intensity] = math.nan

    # Return a logical array
    return matrix == arr


def peak_picking_fpm(matrix, min_distance, threshold_rel):
    """
        Peak picking based on the method describe in Fundamentals of
        Music Processing.

                Parameters:
                    matrix (np.ndarray):
                    min_distance (int):
                    threshold_rel (float):
    """
    # Calculate minimum intensity threshold based on the maximum value in the matrix
    min_intensity = threshold_rel * np.amax(matrix)

    # Calculate neighborhood size
    size = min_distance * 2 + 1

    # If one of the dimensions is not divisible by the neighborhood
    # size add padding to the edges of the grid.
    remainder = np.asarray(matrix.shape) % size
    pad_width = np.where(remainder > 0, size - remainder, 0).astype(int)

    # Add zero padding to the right and bottom of the grid if necessary
    grid = np.pad(matrix, ((0, pad_width[0]), (0, pad_width[1])))

    w, h = grid.shape
    row, col = np.indices((size, size))

    for x, y in itertools.product(range(0, w, size), range(0, h, size)):
        m = grid[row + x, col + y]

        # Find the maximum value in the neighborhood
        maximum = np.amax(m)

        # Assign nan to any value lower than the minimum intensity
        m[m < min_intensity] = math.nan

        # Assign truth values to all the values in the neighborhood
        grid[row + x, col + y] = m == maximum

    rows_to_delete = slice(grid.shape[0] - pad_width[0], grid.shape[0])
    cols_to_delete = slice(grid.shape[1] - pad_width[1], grid.shape[1])

    grid = np.delete(grid, rows_to_delete, axis=0)
    grid = np.delete(grid, cols_to_delete, axis=1)

    return grid


# Investigate different peak picking options for creating constellation maps,
# and use the two lab recordings to plot the derived constellation maps.

# Load the query recording
snd, sr = librosa.load('data/jazz.00005-snippet-10-0.wav')

# Detect peaks from STFT spectrogram and plot constellation maps
D = np.abs(librosa.stft(snd, n_fft=1024, window='hann', win_length=1024, hop_length=512))

# Method from Skimage
coordinates = peak_local_max(np.log(D), min_distance=10, threshold_rel=0.05, indices=False)

plt.figure(figsize=(10, 5))
plt.imshow(coordinates, cmap=plt.get_cmap('gray_r'), origin='lower')
plt.show()

# Method using ndimage.maximum_filter()
coordinates = peak_picking_max_filter(D, min_distance=10, threshold_rel=0.05)

plt.figure(figsize=(10, 5))
plt.imshow(coordinates, cmap=plt.get_cmap('gray_r'), origin='lower')
plt.show()

# Implementation of technique described in Fundamentals of Music Processing
coordinates = peak_picking_fpm(D, min_distance=10, threshold_rel=0.05)

plt.figure(figsize=(10, 5))
plt.imshow(coordinates, cmap=plt.get_cmap('gray_r'), origin='lower')
plt.show()

# Experiment with different time-frequency representations beyond the STFT,
# e.g. you can use the CQT spectrogram or mel spectrogram as alternatives.

# Mel spectrogram and technique described in Fundamentals of Music Processing.
S = librosa.feature.melspectrogram(snd, sr=sr, n_fft=1024, window='hann', win_length=1024, hop_length=512)
coordinates = peak_picking_fpm(S, min_distance=10, threshold_rel=0.05)

plt.figure(figsize=(10, 5))
plt.imshow(coordinates, cmap=plt.get_cmap('gray_r'), origin='lower')
plt.show()

# CQT and technique described in Fundamentals of Music Processing.
C = np.abs(librosa.cqt(snd, sr=sr, window='hann', hop_length=512))
coordinates = peak_picking_fpm(C, min_distance=10, threshold_rel=0.05)

plt.figure(figsize=(10, 5))
plt.imshow(coordinates, cmap=plt.get_cmap('gray_r'), origin='lower')
plt.show()



