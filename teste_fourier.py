from scipy.fft import fft, ifft
from scipy.signal import welch
import wave
import numpy as np
import matplotlib.pyplot as plt


class audio:

    def __init__(self, audiopath):
        '''
        A inicialização do código consiste em definir o caminho do áudio,
        gerar um nome usando manipulação de strings, abrir o arquivo de áudio
        e chamar a transformada de Fourier
        '''

        self.audiopath = audiopath #caminho do áudio
        self.file_name = audiopath[audiopath.rfind('/') + 1:audiopath.rfind('.')] #gera um nome usando manipulação de string
        self.open_audio() #abre o arquivo de áudio
        self.transformed_data = self.calculate_fourier()  #chama a transformada logo na initialização
        #new_name = 'fourier_' + self.file_name #definindo um novo nome

    def open_audio(self):
        '''
        Arquivo criado na forma "wave_write"
        '''

        self.audio_file = wave.open(self.audiopath, 'rb') 

    def calculate_fourier(self):
        '''
        Primeiro o arquivo é lido e convertido para um array do numpy
        Após isso, a transformada de fourrier é feita usando scipy
        '''

        frames = self.audio_file.readframes(-1) #lê o arquivo
        signal = np.frombuffer(frames, dtype='int16') #gera um array do numpy
        transformed_signal = fft(signal) #transformada usando o scipy
        return transformed_signal


###


class denoise(audio):
    '''
    O que será feito nessa função, será a utilização dos dados da transformada
    de Fourier para aplicar o filtro de Wiener e depois voltar os dados para o
    domínio do tempo. Esse filtro foi esolhido porque é o que minimiza os mínimos
    quadrados do modelo. 
    '''

    def __init__(self, audiopath):
        if isinstance(audiopath, str):  # Ensure 'audiopath' is a string
            super().__init__(audiopath)
            self.frequencies, self.psd = self.calculate_psd()
            self.noise = self.calculate_noise(self.psd)
        else:
            raise ValueError("A file path string is required.")


    #Calcular o psd a ser usado no cáluclo do filtro de Wiener
    def calulate_psd(self):
        frequencies, psd = welch(self.transformed_data.real, fs=self.audio_file.getframerate())
        return frequencies, psd

    #o threshols está ajustado em 15%
    def calculate_noise(self, noise_psd, threshold=0.15):
        noise = np.where(noise_psd > threshold * np.max(noise_psd))[0]
        return noise
        

    def wiener_filter(self):
        wiener_filter = self.psd / (self.psd + self.noise) #cálculo do filtro

        filtered_data = self.transformed_data * wiener_filter #aplicando o filtro
        denoised_signal = ifft(filtered_data) #transformada inversa
        return denoised_signal


###


class plot(audio):
    '''
    Essa classe visa a criação dos gráficos do áudio original e da transformada de Fourrier realizada.
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
    def plot_filtered_signal(self, filtered_data):
        plt.figure(figsize=(8, 6))
        plt.plot(filtered_data)
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.title('Filtered Signal')
        plt.show()

    def plot_inverse_fft(self, denoised_signal):
        plt.figure(figsize=(8, 6))
        plt.plot(denoised_signal)
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.title('Inverse Fourier Transform')
        plt.show()






##################


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