import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
import plotly.graph_objs as go
from scipy.io import wavfile

class audio_denoise:
    def __init__(self, path_to_file):
        # Load audio file saving the floating points y and audio rate
        self.y, self.sr = librosa.load(path_to_file, sr = None)
        # Get time values from number of floating points intensity divided by the audio rate
        self.time = np.arange(0, len(self.y))/ self.sr
        # Get file name to name the output file later on
        self.file_name = path_to_file[path_to_file.rfind('/') + 1:path_to_file.rfind('.')]
        # Start denoised signal variable
        self.denoised_signal = None
        # Start the used transform name to name the output file later on
        self.transform = None

    
    # Performs a 1 dimension wavelet transform with standard parameters if none are indicated
    def wavelet_transform_1D(self, threshold, wavelet='db1', mode ='sym', level=None):
        # Perform a multi-level wavelet decomposition
        coeffs = pywt.wavedec(self.y, wavelet, mode=mode, level=level)
        # Set a threshold to nullify smaller coefficients assumed to be noise
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        # Reconstruct the signal from the thresholded coefficients
        self.denoised_signal = pywt.waverec(coeffs_thresholded, 'db1')
        # Used transform identifier
        self.transform = 'wavelet_'

    # Writes transformed signal into an audio file
    def file_write(self):
        # Name the file with it's extension
        output_file = self.transform + self.file_name + ".wav"
        # Writes file with the output_file name
        wavfile.write(output_file, int(self.sr), self.denoised_signal)

    # Testing function to check if the usage of different natured libraries (librosa for loading and scipy.io.wavfile for writing) 
    # in the audio reconstruction would insert any noise or loss of quality - (answer: No quality loss was noticed)
    def test_audio_recon(self):
        output_file = "reconstruction_TEST_" + self.file_name + ".wav"
        wavfile.write(output_file, int(self.sr), self.y)

    # Plots transform graphs for comparison
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





##################

if __name__ == '__main__':
    audio_file = "your_audio_file.wav"

    denoising = audio_denoise(audio_file)

    denoising.wavelet_transform_1D(0.005)
    denoising.file_write()
