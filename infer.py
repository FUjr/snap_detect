import wave
import librosa
import numpy as np
import pyaudio
import tensorflow as tf
import time
from tqdm import tqdm
# 获取网络模型
model = tf.keras.models.load_model('voice_classfication.h5')

# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "infer_audio_%s.wav"

model = tf.saved_model.load('voice_model')

# 打开录音
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)



# 获取录音数据
def record_audio():
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    npframes = np.frombuffer(b''.join(frames), dtype=np.int16)
    #补齐16000
    if len(npframes) < 16000:
        npframes = np.pad(npframes, (0, 16000-len(npframes)), 'constant')
    #映射到[-1,1]
    npframes = npframes / 32768.0
    #修改dtype
    npframes = npframes.astype(np.float32)
    npframes = npframes[tf.newaxis, :]
    return npframes


# 预测
def infer():
    #load model "voice_classfication.h5"
    model = tf.saved_model.load('voice_model')
    result = model(record_audio())
    #获取预测结果
    for f in result:
        print(f)
        print(result['class_names'].numpy()[0].decode('utf-8'))
        print(result['predictions'].numpy())
        print(result['class_ids'].numpy())
        break
    


if __name__ == '__main__':
    try:
        while True:
            # 获取预测结果
            infer()

    except Exception as e:
        print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()
