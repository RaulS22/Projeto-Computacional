from scipy.fft import fft
import wave
import numpy as np
import matplotlib.pyplot as plt


class audio:
    '''
    Nessa classe o arquivo de áudio será tratado de
    forma a ser entendido pelo resto do código. Serão
    usados recursos de algumas bibliotecas.
    '''

    def __init__(self, audiopath):
        audio_file = wave.open(audiopath, 'rb')
        self.file_name = audiopath[audiopath.rfind('/') + 1:audiopath.rfind('.')]
        # novo_nome =  + self.file_name #TODO:implementar a mudança de nome após plotar o gráfico e gerar o arquivo
        
        signal = audio_file.readframes(-1)
        #signal = np.fromstring(signal, 'Int16')


    def fourier(self):
        pass
        #TODO: implementar a transformação de Fourier



    def plots(self):
        pass
        #TODO: implementar a classe de plotagem dos gráficos


    '''
    Aqui o arquivo de áudio está sendo criado da forma
    "wave_write"
    A variável signal recebe uma array com o conteúdo do
    áudio
    '''

    







################################
'''
Referências:
https://docs.python.org/3/library/wave.html
https://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file
'''