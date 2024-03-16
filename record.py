import pyaudio
import wave
import time
from tqdm import tqdm
import os

AUDIO_TYPE = input('Input class：(noise/snap)')
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
AUDIO_DIR = "saved_audio"
WAVE_OUTPUT_FILENAME = os.join.path(AUDIO_DIR,AUDIO_TYPE,'%s_%s.wav'% (AUDIO_TYPE,"%s"))

if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)
if not os.path.exists(os.path.join(AUDIO_DIR,AUDIO_TYPE)):
    os.makedirs(os.path.join(AUDIO_DIR,AUDIO_TYPE))
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
while True:
    frames = []
    for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
        data = stream.read(CHUNK)
        frames.append(data)
    fileNmae = WAVE_OUTPUT_FILENAME  % str(time.time())
    wf = wave.open(fileNmae, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print('error File exist：%s' % fileNmae)
