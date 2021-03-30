# def peak_picking(signal, algorithm='max_filter'):
#     """
#         Detect peaks from an STFT spectrogram
#
#             Parameters:
#                 signal (np.ndarray):
#                 algorithm (string):
#     """
#     matrix_d = np.abs(librosa.stft(y, n_fft=1024, window='hann', win_length=1024, hop_length=512))
#
#     if algorithm == 'max_filter':
#         threshold_rel = 0.05
#         min_intensity = threshold_rel * np.amax(D)
#
#         arr = ndimage.maximum_filter(D, size=20)
#         arr[arr < min_intensity] = math.nan