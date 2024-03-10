import pyaudio
import wave
import uuid
from tqdm import tqdm
import os


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "save_audio/snap_%s.wav"
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
if not os.path.exists('save_audio'):
    os.makedirs('save_audio')
while True:
    frames = []
    for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
        data = stream.read(CHUNK)
        frames.append(data)
    fileNmae = WAVE_OUTPUT_FILENAME  % str(uuid.uuid1()).replace('-', '')
    wf = wave.open(fileNmae, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print('文件保存在：%s' % fileNmae)
