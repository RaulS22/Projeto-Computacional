import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
import numpy as np
import pywt
import plotly.graph_objs as go
import pandas as pd
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import welch, wiener


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
        self.denoised_signal = self.y # Start denoised signal variable
        self.transform = 'none_' # Start the used transform name to name the output file later on
        self.file_name = 'data' # Start file name in case of usage of the audio_write func

        # If a noise file path is provided, save its floating points to a noise variable (this applies to the noise_reduce() function)
        if type(noise_data) != bool:
            self.noise = np.array(noise_data).astype(float)
        else:
            self.noise = noise_data

    
    def wavelet_transform_1D(self, threshold, wavelet='db1', mode ='sym', level=None):
        '''
        Performs a 1 dimension multilevel wavelet transform with standard parameters if none are indicated,
        decomposing the signal, filtering and reconstructing it.

        Parameters:
        - threshold: signal intensity filtering threshold; atenuates lower intensity signal bits
        over wavelet's transform time windows, highlighting peak values.

        - wavelet: sets which wavelet is used to decompose the signal (wavelet standard used if none are provided: 'db1').
        Different wavelets can be used in signal decomposition and the PyWavelets package docs wields
        more information on all the wavelets available and its respective string codes (type pywt.wavelist or check https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html).

        - mode: reffers to the extrapolation mathematical tool for signal extension (standard: 'sym', short term for symmetric)
        Different extrapolations can perform different kinds of artifacts in the ends of the signal,
        see PyWavelt documentation for more information (https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes).

        - level: reffers to the level of the multilevel transform, if none are provided it is calculated using dwt_max_level.

        Returns: 

        - self.denoised_signal: after the signal in decomposed and filtered, it is reconstructed to a denoised version.
        
        (check PyWavelets Multilevel DWT docs)
        '''
        self.transform = 'wavelet_' # Used transform identifier
        coeffs = pywt.wavedec(self.y, wavelet, mode=mode, level=level)
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs] # Set a threshold to nullify smaller coefficients assumed to be noise
        self.denoised_signal = pywt.waverec(coeffs_thresholded, 'db1') # Reconstruct the signal from the thresholded coefficients

     
    def noise_reduce(self):
        '''
        Uses de reduce_noise function from the noisereduce lib. It can either
        apply noise reduction from noise file (if provided) or apply noise 
        reduction with estimated noise profile.

        Parameters:


        Returns: 
        -
        '''

        if type(self.noise) != bool:
            self.denoised_signal = nr.reduce_noise(y=self.y, sr=self.sr, y_noise=self.noise)
        else:
            self.denoised_signal = nr.reduce_noise(y=self.y, sr=self.sr)
        self.transform = 'nr_'

    def fourier_transform(self):
        '''
        Performs a FFT, uses the wiener filter and then performs an Inverse FFT.

        Parameters:

        Returns: out: 1darray
                 Denoised data as the result of the fft, filtering and ifft process
        '''

        self.transform = 'fourier_'
        transformed_signal = fft(self.y) # Performs FFT on the original signal

        abs_t_sig = abs(transformed_signal) # Gets absolute values of frequency amplitudes
        filtered_data = wiener(abs_t_sig) # Performs a Wiener filter on the absolute values
        filtering_weights = filtered_data / abs_t_sig # Gets the weights applied to each floating point on the wiener filter
        filtered_signal = filtering_weights*transformed_signal # Applies the weights on the complex amplitudes
        denoised_abs_signal = abs(ifft(filtered_signal)) # Performs and IFFT on the filtered signal, which returns complex values
        #                                                  due to approximations and filtering made before, and extract it's absolute values

        y = self.y      # Necessary lines to make the calculations to retrieve information on the signs of
                        # the amplitude values of the original signal.
        y[y == 0] = 1   # Getting rid of possible 0/0 indeterminations 
        amplitude_signs = y / abs(self.y) # Retrieving the amplitude signs

        self.denoised_signal = denoised_abs_signal*amplitude_signs # Applying respective positive and negative signs on the filtered signal's
                                                                   # complex "floating" points absolute values to retrieve oscilations
        
        self.denoised_signal = np.float32(self.denoised_signal) # Formmating the denoised signal to float32 dtype for compatilibity with the
                                                                # audio_write function.

    def plot_denoising(self, comp=True, sep=False):
        '''
        Plots three figures: one with the comparassion of original and denoised data, the other
        only with the original data and the last one with the denoised data.

        Parameters: comp: bool
                        If  comp == True, the comparison image will be ploted
                    sep: bool
                        If  sep == True, both the original and the denoised image will be ploted

        Returns:
            Figures
        '''    
        
    
        plt.figure(figsize=(10, 6))
        
        if comp:
            plt.plot(self.time, self.y, label='Original')
            plt.plot(self.time, self.denoised_signal, label='Denoised')
            plt.title('Comparassion between original signal and denoised signal ' + self.file_name)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()
        
        if sep:
            plt.figure(figsize=(10, 6))
            plt.plot(self.time, self.y, label='Original')
            plt.title('Original signal of ' + self.file_name)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(self.time, self.denoised_signal, label='Denoised')
            plt.title('Denoised signal of ' + self.file_name)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()
    



class audio_denoise(data_denoise):

    def __init__(self, path_to_file, noise_file_path = False):
        '''
        Load audio file saving the floating points y and audio rate


        Parameters: path_to_file: string
                    A path to the audio file

                    noise_file_path: string
                    A path to the noise file (if there is one)


        Returns:    out: .wav file
                    It returns an .wav denoised file
        
        '''

        if isinstance(path_to_file, str):
        
            self.y, self.sr = librosa.load(path_to_file, sr = None) # Get time values from number of floating points intensity divided by the audio rate
            self.time = np.arange(0, len(self.y))/ self.sr # Get file name to name the output file later on
            self.file_name = path_to_file[path_to_file.rfind('/') + 1:path_to_file.rfind('.')] # Start denoised signal variable
            self.denoised_signal = self.y # Start the used transform name to name the output file later on
            self.transform = 'none'

            # If a noise file path is provided, save its floating points to a noise variable (applies to noise_reduce() function)
            if type(noise_file_path) != bool:
                self.noise, _ = librosa.load(noise_file_path, sr=None)
            else:
                self.noise = noise_file_path

        else:
            raise ValueError("A file path string is required.")


    def audio_write(self):
        '''
        Writes transformed signal into an audio file


        Parameters:


        Returns: 
        -
        '''

        
        output_file = self.transform + self.file_name + ".wav" # Name the file with it's extension
        wavfile.write(output_file, int(self.sr), self.denoised_signal) # Writes file with the output_file name

if __name__ == "__main__":
    audio_file = 'noisy-sample-1.wav'
    denoising = audio_denoise(audio_file)

    denoising.fourier_transform()
    denoising.audio_write()


