from scipy.fft import fft
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

audio = 'FazOL.wav' #caso não reconheça, ou use o caminho do áudio, ou mude o path


'''
O que está acontecendo na sexta linha é que
audio_data está recebendo a informação do áudio em forma de array
freq recebe a frequência (em Hz)
'''

