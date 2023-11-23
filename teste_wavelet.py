import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
import plotly.graph_objs as go
from scipy.io import wavfile

class audio_denoise:
    def __init__(self, path_to_file):
        # Load audio file
        self.y, self.sr = librosa.load(path_to_file)
        # Get time values
        self.time = np.arange(0, len(self.y))/ self.sr
        # Get file name for naming the output
        self.file_name = path_to_file[path_to_file.rfind('/') + 1:path_to_file.rfind('.')]
        # Start denoised signal variable
        self.denoised_signal = None


    def wavelet_transform(self, threshold):
        # Perform a multi-level wavelet decomposition
        coeffs = pywt.wavedec(self.y, 'db1', level=4)
        # Set a threshold to nullify smaller coefficients (assumed to be noise)
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        # Reconstruct the signal from the thresholded coefficients
        self.denoised_signal = pywt.waverec(coeffs_thresholded, 'db1')
        self.denoised_signal = np.array(self.denoised_signal)
        output_file = "wavelet_" + self.file_name + ".wav"
        wavfile.write(output_file, int(self.sr), self.denoised_signal)

    def plot_denoising(self):
        fig = go.Figure()

        fig.add_scatter(x=self.time, y=self.y,  mode='lines', name = 'Original')
        fig.add_scatter(x=self.time, y=self.denoised_signal, mode='lines', name = 'Denoised')

        fig.update_layout(
            title='Denoised signal of ' + self.file_name + ' comparison',
            xaxis_title='Time',
            yaxis_title='Amplitude',
            
        )
        fig.show()



audio_file = "FazOL.wav"

denoising = audio_denoise(audio_file)

denoising.wavelet_transform(0.01)
