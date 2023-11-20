from scipy.fft import fft
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

audio = 'FazOL.wav' #caminho do áudio TODO: arrumar isso
audio_data, freq = sf.read(audio)

'''
O que está acontecendo na sexta linha é que
audio_data está recebendo a informação do áudio em forma de array
freq recebe a frequência (em Hz)
'''

y = fft(audio_data) #transformada de Fourrier do audio

