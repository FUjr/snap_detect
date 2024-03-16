import numpy as np
import pyaudio
import tensorflow as tf
import json,os,sys,time
import matplotlib.pyplot as plt
import threading
from homeassistant_api import Client
url = 'http://192.168.1.119:8123/api/'
token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI0ZDg2YTU5ZDkzNzc0ODIwYWM4YmU3ZDljMGFjYmE4NSIsImlhdCI6MTcxMDI0NzU3OCwiZXhwIjoyMDI1NjA3NTc4fQ.kRIG_keHrL-Sx8Qw8Ywzdo3nmXBJS6tnJt6Ke2QQLHw'
# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
args = sys.argv
frame_lock = threading.Lock()
if len(args) == 4:
    index = args[1]
    Network_type = args[2]
    Format = args[3]
else:
    index = input('请输入模型编号：')
    Network_type = input('请输入网络类型：')
    Format = input('请输入格式：')
model_name = os.path.join('model_%s' % index, 'model_%s.%s' % (Network_type,Format))
config_name = os.path.join('model_%s' % index, 'model_info_%s.json' % Network_type)
config = json.load(open(config_name))
BEST_THRESHOLD = config['best_threshold']
RATE = 16000
NUM_MEL_BINS = config['num_mel_bins']
LOWER_EDGE_HERTZ = config['lower_edge_hertz']
UPPER_EDGE_HERTZ = config['upper_edge_hertz']
FRAME_LENGTH = config['frame_length']
FRAME_STEP = config['frame_step']
RECORD_SECONDS = 1
EXPORT = 0
if Format == 'export':
    model = tf.saved_model.load(model_name)
    EXPORT = 1
else:
    model = tf.keras.models.load_model(model_name,compile=False)

# 打开录音


def convert2mel(audio, sample_rate=RATE,
                num_mel_bins=NUM_MEL_BINS, lower_edge_hertz=LOWER_EDGE_HERTZ, upper_edge_hertz=UPPER_EDGE_HERTZ):
    # 计算 STFT
    stfts = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)
    
    # 获取频谱幅度
    spectrograms = tf.abs(stfts)
    
    # 计算梅尔权重矩阵
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    
    # 将频谱转换为梅尔频谱
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    #增加batch维度
    mel_spectrograms = tf.expand_dims(mel_spectrograms, 0)
    return mel_spectrograms


# 
def record_audio():
    global chunks
    chunks = []
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    try:
        while True:
            time.sleep(0.001)
            data = stream.read(CHUNK)
            if len(data) == 0:
                continue
            frame_lock.acquire()
            chunks.append(data)
            frame_lock.release()
    except Exception as e:
        print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()


def export_model_infer(raw_audio):
    raw_audio = np.expand_dims(raw_audio, axis=0)
    return model(raw_audio)

# 预测
def keras_infer(raw_audio):
    mel = convert2mel(raw_audio)
    return model(mel,training=False)

def get_infer():
    if EXPORT:
        return export_model_infer
    else:
        return keras_infer


def do(result,window_length):
    global chunks
    result = result.numpy()
    if result[0][0] > BEST_THRESHOLD:
        print('响指置信度为：', result[0][0])
        with Client(url, token) as client:
            light = client.get_domain('light')
            light.toggle(entity_id = 'light.yeelink_lamp22_4f32_light')
        #plt展示mel
        frame_lock.acquire()
        chunks = chunks[window_length:]
        frame_lock.release()
    else:
        #张量转numpy
        print(result[0][0])


def main():
    global chunks
    infer = get_infer()
    window_step_seconds = 0.5
    window_seconds = 1
    simple_rate = 16000
    window_step = int(window_step_seconds * simple_rate / CHUNK)
    window_length = int(window_seconds * simple_rate / CHUNK) + 1
    last_infer = time.time()
    while True:
        time.sleep(0.1)
        frame_lock.acquire()
        if len(chunks) < window_length:
            frame_lock.release()
            continue
        else:
            audio = chunks[:window_length]
            chunks = chunks[window_step:]
            frame_lock.release()
            audio = np.frombuffer(b''.join(audio), dtype=np.int16) / 32768
            audio = audio[:window_seconds * simple_rate]
            audio = np.array(audio, dtype=np.float32)
            result = infer(audio)
            do(result,window_length)


if __name__ == '__main__':
    
        while True:
            record_audio = threading.Thread(target=record_audio,daemon=True).start()
            main()
            record_audio.stop()

    
