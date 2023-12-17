from scipy.fft import fft, ifft
from scipy.signal import welch
from scipy import signal
import wave
import numpy as np
import matplotlib.pyplot as plt


class audio:

    def __init__(self, audiopath):
        '''
        Checks if a string was given. If don't, it will raise an error.
        '''
        if isinstance(audiopath, str):
            self.audiopath = audiopath #audio path
            self.file_name = audiopath[audiopath.rfind('/') + 1:audiopath.rfind('.')] #generates a name using string manipulation
            self.open_audio() #opens the audio file
            self.transformed_data = self.calculate_fourier()  #call the transform at the init
            
            #new_name = 'fourier_' + self.file_name #definindo um novo nome
        else:
            raise ValueError("A file path string is required.")

    def open_audio(self):
        '''
        File created at the "wave_write" mode
        '''

        self.audio_file = wave.open(self.audiopath, 'rb') 

    def calculate_fourier(self):
        '''
        First, the file is read and converted to an numpy array
        After it, a Fourier transform is done using Scipy
        '''

        frames = self.audio_file.readframes(-1) #reads the file
        signal = np.frombuffer(frames, dtype='int16') #generates an numpy array
        transformed_signal = fft(signal) #Fourier transform using scipy
        return transformed_signal


###


class denoise(audio):
    '''
    What is being done in this function is the use of the Fourier Transform data
    to apply the Wiener filter and, after it, apply the Inverse Fourier Transform
    to return the data to the time domain. This filter was choosen because it is
    the method that minimizes the mean square error of the model.
    '''

    def __init__(self, audiopath):
        super().__init__(audiopath)
        self.frequencies, self.psd = self.calculate_psd()
        self.noise = self.calculate_noise(self.psd)
        


    #Calculates the psd to be used at the Wiener filter calculus
    def calculate_psd(self):
        frequencies, psd = welch(self.transformed_data.real, fs=self.audio_file.getframerate())
        return frequencies, psd

    #nsd -> noise spectral density
    def calculate_nsd(self):
        frequencies, noise_psd = signal.periodogram(self.transformed_data.real, fs=self.audio_file.getframerate())
        return noise_psd


    def calculate_noise(self, noise_psd, threshold=0.15):
        '''
        The threshols choosen was 0.15. This may require adjustments deppending on 
        your audio file
        '''
        noise = np.where(noise_psd > threshold * np.max(noise_psd))[0]
        return noise
    
        

    def wiener_filter(self):
        wiener_filter = self.psd / (self.psd + self.noise) #Wiener filter calculus

        filtered_data = self.transformed_data * wiener_filter #applying the filter
        denoised_signal = ifft(filtered_data) #invese fourier transform
        return denoised_signal
    
    


###


class plot(audio):
    '''
    This class aims the graphs creation. It can generate the original file, its 
    Fourier Transform, the data after the filter and the denoised audio. OpenAI 
    Chat GPT was used to create some of the graphs.
    '''

    def __init__(self, audiopath):
        self.audiopath = audiopath
        self.audio_file = wave.open(self.audiopath, 'rb')
        self.transformed_data = self.calculate_fft()

    def calculate_fft(self):
        frames = self.audio_file.readframes(-1)
        signal = np.frombuffer(frames, dtype='int16')
        return np.fft.fft(signal)


    def plot_fft(self):
        transformed_signal = self.transformed_data
        freq = np.fft.fftfreq(len(transformed_signal)) * self.audio_file.getframerate()

        plt.figure(figsize=(8, 6))
        plt.plot(freq, np.abs(transformed_signal))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Fourier Transform')
        plt.show()

    #TODO: Fazer o áudio ficar visível
    def plot_audio(self):
        frames = self.audio_file.readframes(-1)
        signal = np.frombuffer(frames, dtype='int16')

        plt.figure(figsize=(8, 6))
        plt.imshow(signal.reshape(-1, 1).T, aspect='auto', cmap='viridis')
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.title('Audio Signal')
        plt.colorbar(label='Amplitude')
        plt.show()

    #TODO: Implementar o gráfico da transformada com o filtro e do áudio tratado
    





################################

if __name__ == '__main__':

    #Testando a transformada
    audio_path = 'teste-blastoise.wav'
    audio_instance = audio(audio_path)
    print(f"O nome do arquivo é: {audio_instance.file_name}")
    print(f"Os dados são: {audio_instance.transformed_data}")

    #Testando os gráficos
    audio_plot = plot(audio_path)
    audio_plot.plot_audio() 
    audio_plot.plot_fft()

    #TODO: testar o filtro e o cancelamento de ruído
    



################################

'''
Referências:
https://docs.python.org/3/library/wave.html
https://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file
https://ocw.mit.edu/courses/6-011-introduction-to-communication-control-and-signal-processing-spring-2010/f135b328c7448bf21c4939ea9ff8f8fb_MIT6_011S10_chap11.pdf


I used OpenAI ChatGPT to do the graphs
I also used it in order to fix some errors and to organize my code in a better way
'''