import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
import numpy as np
import pywt
import plotly.graph_objs as go
import pandas as pd
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import welch
from scipy import signal
import wave

class data_denoise():
    
    def __init__(self, y_data, x_data = False, noise_data = False):
        '''
        This function deals with data in a x and y-axis. If x data is not provided,
        an array will be created with equal intervals of time (1, 2, 3, ...)

        Parameters:


        Returns: 
        -
        '''
        self.y = np.array(y_data).astype(float)
        if type(x_data) != bool:
            self.time = np.array(x_data).astype(float)
        else:
            self.time = np.arange(0, len(self.y))
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

    
    def wavelet_transform_1D(self, threshold, wavelet='db1', mode ='sym', level=None):
        '''
        Performs a 1 dimension wavelet transform with standard parameters if none are indicated

        Parameters:


        Returns: 
        -
        '''
        coeffs = pywt.wavedec(self.y, wavelet, mode=mode, level=level)
        # Set a threshold to nullify smaller coefficients assumed to be noise
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        # Reconstruct the signal from the thresholded coefficients
        self.denoised_signal = pywt.waverec(coeffs_thresholded, 'db1')
        # Used transform identifier
        self.transform = 'wavelet_'

     
    def noise_reduce(self):
        '''
        Uses de reduce_noise function from the noisereduce lib. It can either
        apply noise reduction from noise file (if provided) or apply noise 
        reduction with estimated noise profile

        Parameters:


        Returns: 
        -
        '''
        if type(self.noise) != bool:
            self.denoised_signal = nr.reduce_noise(y=self.y, sr=self.sr, y_noise=self.noise)
        else:
            self.denoised_signal = nr.reduce_noise(y=self.y, sr=self.sr)
        self.transform = 'nr_'




class audio_denoise(data_denoise):
    def __init__(self, path_to_file, noise_file_path = False):
        '''
        Load audio file saving the floating points y and audio rate


        Parameters:


        Returns: 
        -
        '''
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


    def audio_write(self):
        '''
        Writes transformed signal into an audio file

        
        Parameters:


        Returns: 
        -
        '''

        # Name the file with it's extension
        output_file = self.transform + self.file_name + ".wav"
        # Writes file with the output_file name
        wavfile.write(output_file, int(self.sr), self.denoised_signal)