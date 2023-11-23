from scipy.fft import fft, ifft
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


class plot(audio):
    '''
    Essa classe visa a criação dos gráficos do áudio original e da transformada de Fourrier realizada.
    '''

    def plot_audio(self):
        audio_file = wave.open(self.audiopath, 'rb')
        frames = self.audio_file.readframes(-1)
        signal = np.frombuffer(frames, dtype='int16')

        plt.figure(figsize=(8, 6))
        plt.imshow(signal.reshape(-1, 1).T, aspect='auto', cmap='viridis')
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.title('Audio Signal')
        plt.colorbar(label='Amplitude')
        plt.show()

    def plot_fft(self):
        audio_file = wave.open(self.audiopath, 'rb')
        transformed_signal = self.transformed_data
        freq = np.fft.fftfreq(len(transformed_signal)) * self.audio_file.getframerate()

        plt.figure(figsize=(8, 6))
        plt.plot(freq, np.abs(transformed_signal))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Fourier Transform')
        plt.show()



##################

#Testando os gráficos
if __name__ == '__main__':
    audio_path = 'teste-blastoise.wav'
    audio_plot = plot(audio_path)
    audio_plot.plot_audio()
    audio_plot.plot_fft()


################################
'''
Referências:
https://docs.python.org/3/library/wave.html
https://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file

I also used Open AI ChatGPT in order to fix some errors and to organize my code in a better way
'''