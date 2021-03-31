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


def plot_constellation_map(coords):
    """
        Plot a constellation map based on the coordinates of the nonzero elements in
        the logical array.  A scatter plot or pyplot's spy() function can be used.
        The imshow() function produces a visualisation that is too small to be useful.

        Parameters:
            coords (np.ndarray):

    """
    plt.figure(figsize=(10, 5))
    plt.imshow(coords, cmap=plt.get_cmap('gray_r'), origin='lower')

    y, x = np.nonzero(coords)
    plt.scatter(x, y, s=10, marker='.', c='black')

    # Can use either spy() or a scatter()
    # plt.spy(coordinates, marker='.', markersize=5, mfc='black', mec='black', origin='lower')

    plt.show()


def fingerprint_matching(query, database_recording):
    rows, cols = np.indices(query.shape)

    # Shifting the query by m positions yields the constellation map m + C(Q)
    m_positions = database_recording.shape[1] - query.shape[1] + 1

    # As long as the number of correctly matching
    # peak coordinates is statistically significant
    percentages = np.zeros(m_positions)

    for m in range(m_positions):
        result = query == database_recording[rows, cols + m]
        percentages[m] = np.sum(result) / query.size

    index_of_maximum = np.argmax(percentages)
    match = database_recording[rows, cols + index_of_maximum]
    print(np.amax(percentages))

    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(111)
    #
    # y, x = np.nonzero(database_recording)
    # ax.scatter(x, y, s=10, marker='.', c='black')
    #
    # y, x = np.nonzero(match)
    # ax.scatter(x, y, s=1, marker='.', c='red')

    plt.figure(figsize=(10, 5))

    y, x = np.nonzero(database_recording)
    plt.scatter(x, y, s=10, marker='.', c='black')

    y, x = np.nonzero(match)
    plt.scatter(x + index_of_maximum, y, s=1, marker='.', c='red')

    plt.show()

    # plot_constellation_map(match)
    # plot_constellation_map(database_recording)


def spectrogram(path):
    y, sr = librosa.load(path)

    # STFT returns a complex-valued matrix D so we need
    # to take the absolute value of each element.
    return np.abs(librosa.stft(y, n_fft=1024, window='hann', win_length=1024, hop_length=512))


# Investigate different peak picking options for creating constellation maps,
# and use the two lab recordings to plot the derived constellation maps.

spec = spectrogram('data/jazz.00005-snippet-10-0.wav')
CQ = peak_local_max(np.log(spec), min_distance=10, threshold_rel=0.05, indices=False)

print('CQ Dimensions: ', CQ.shape)
plot_constellation_map(CQ)

p = 'data/jazz.00005.wav'
CD = peak_local_max(np.log(spectrogram(p)), min_distance=10, threshold_rel=0.05, indices=False)

print('CD Dimensions: ', CD.shape)
plot_constellation_map(CD)

fingerprint_matching(CQ, CD)

# # Method using ndimage.maximum_filter()
# coordinates = peak_picking_max_filter(D, min_distance=10, threshold_rel=0.05)
# plot_constellation_map(coordinates)
#
# # Implementation of technique described in Fundamentals of Music Processing
# coordinates = peak_picking_fpm(D, min_distance=10, threshold_rel=0.05)
# plot_constellation_map(coordinates)
#
# # Experiment with different time-frequency representations beyond the STFT,
# # e.g. you can use the CQT spectrogram or mel spectrogram as alternatives.
#
# # Mel spectrogram and technique described in Fundamentals of Music Processing.
# S = librosa.feature.melspectrogram(snd, sr=sr, n_fft=1024, window='hann', win_length=1024, hop_length=512)
#
# coordinates = peak_picking_fpm(S, min_distance=10, threshold_rel=0.05)
# plot_constellation_map(coordinates)
#
# # CQT and technique described in Fundamentals of Music Processing.
# C = np.abs(librosa.cqt(snd, sr=sr, window='hann', hop_length=512))
#
# coordinates = peak_picking_fpm(C, min_distance=10, threshold_rel=0.05)
# plot_constellation_map(coordinates)

# Explore different parameters for defining your peak picking neighbourhood.

# What could these be?  I guess a neighborhood doesn't necessarily have to
# be perfectly square...
