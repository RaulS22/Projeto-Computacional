import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
import numpy as np
import pywt
import plotly.graph_objs as go
import pandas as pd
from scipy.io import wavfile

class data_denoise():
    def __init__(self, y_data, x_data = False, noise_data = False):
        # Load signal data
        self.y = np.array(y_data).astype(float)
        # Load "time" data
        if type(x_data) != bool:
            self.time = np.array(x_data).astype(float)
        else:
            self.time = np.arrange(0, len(self.y))
        # Compute rate
        self.sr = (len(self.y)/self.time[-1])
        # Start denoised signal variable
        self.denoised_signal = self.y
        # Start the used transform name to name the output file later on
        self.transform = 'none_'
        # Start file name in case of usage of the audio_write func
        self.file_name = 'data'
        # If a noise file path is provided, save its floating points to a noise variable (applies to noise_reduce() function)
        if type(noise_data) != bool:
            self.noise = np.array(noise_data).astype(float)
        else:
            self.noise = noise_data
    
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

    # Uses de reduce_noise function from the noisereduce lib, but 
    def noise_reduce(self):
        # Apply noise reduction from noise file (if provided)
        if type(self.noise) != bool:
            self.denoised_signal = nr.reduce_noise(y=self.y, sr=self.sr, y_noise=self.noise)
        # Apply noise reduction with estimated noise profile
        else:
            self.denoised_signal = nr.reduce_noise(y=self.y, sr=self.sr)
        self.transform = 'nr_'

    # Testing function to check if the usage of different natured libraries (librosa for loading and scipy.io.wavfile for writing) 
    # in the audio reconstruction would insert any noise or loss of quality - (answer: No quality loss was noticed)
    def test_audio_recon(self):
        output_file = "reconstruction_TEST_" + self.file_name + ".wav"
        wavfile.write(output_file, int(self.sr), self.y)

    # Plots transform graphs for comparison
    def plot_denoising(self, comp=True, sep=False):
        if comp:
            fig = go.Figure()

            fig.add_scatter(x=self.time, y=self.y,  mode='lines', name = 'Original')
            fig.add_scatter(x=self.time, y=self.denoised_signal, mode='lines', name = 'Denoised')

            fig.update_layout(
                title='Denoised signal of ' + self.file_name + ' comparison',
                xaxis_title='Time',
                yaxis_title='Amplitude',
            )

            fig.show()
        
        if sep:
            fig = go.Figure()

            fig.add_scatter(x=self.time, y=self.y,  mode='lines', name = 'Original')
            fig.update_layout(
                title='Original signal of ' + self.file_name,
                xaxis_title='Time',
                yaxis_title='Amplitude',
            )

            fig.show()

            fig = go.Figure()

            fig.add_scatter(x=self.time, y=self.denoised_signal,  mode='lines', name = 'Original')
            fig.update_layout(
                title='Denoised signal of ' + self.file_name,
                xaxis_title='Time',
                yaxis_title='Amplitude',
            )

            fig.show()

class audio_denoise(data_denoise):
    def __init__(self, path_to_file, noise_file_path = False):
        # Load audio file saving the floating points y and audio rate
        self.y, self.sr = librosa.load(path_to_file, sr = None)
        # Get time values from number of floating points intensity divided by the audio rate
        self.time = np.arange(0, len(self.y))/ self.sr
        # Get file name to name the output file later on
        self.file_name = path_to_file[path_to_file.rfind('/') + 1:path_to_file.rfind('.')]
        # Start denoised signal variable
        self.denoised_signal = self.y
        # Start the used transform name to name the output file later on
        self.transform = 'none'
        # If a noise file path is provided, save its floating points to a noise variable (applies to noise_reduce() function)
        if type(noise_file_path) != bool:
            self.noise, _ = librosa.load(noise_file_path, sr=None)
        else:
            self.noise = noise_file_path

    # Writes transformed signal into an audio file
    def audio_write(self):
        # Name the file with it's extension
        output_file = self.transform + self.file_name + ".wav"
        # Writes file with the output_file name
        wavfile.write(output_file, int(self.sr), self.denoised_signal)



if __name__ == '__main__':
    audio_file = 'sample-1.wav'
    denoising = audio_denoise(audio_file)

    denoising = audio_denoise(audio_file, 'noise-sample-1.wav')
    y = denoising.y
    x = denoising.time
    noise = denoising.noise
