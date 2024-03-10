import scipy.io.wavfile as wavfile
import os
import numpy as np
def split_stereo(file_name):
    # default stereo
    samplerate, data = wavfile.read(file_name)
    left = []
    right = []
    for item in data:
        left.append(item[0])
        right.append(item[1])
    wavfile.write(file_name, samplerate, np.array(left))
dir = 'stereo'
for file in os.listdir(dir):
    if file.endswith('.wav'):
        file = os.path.join(dir, file)
        split_stereo(file)
