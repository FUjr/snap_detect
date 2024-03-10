import pyaudio
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
frames = []
for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
    data = stream.read(CHUNK)
    frames.append(data)
npframes = np.frombuffer(b''.join(frames), dtype=np.int16)
#映射到[-1,1]
npframes = npframes / 32768.0
#转为频谱图
ps= tf.signal.stft(npframes, frame_length=512, frame_step=64)

ps = tf.abs(ps)
ps = ps.numpy()
#显示频谱图
log_spec = np.log(ps.T + np.finfo(float).eps)
height = log_spec.shape[0]
width = log_spec.shape[1]
print(height, width)
X = np.linspace(0, np.size(ps), num=width, dtype=int)
Y = range(height)
plt.pcolormesh(X, Y, log_spec)
plt.show()
