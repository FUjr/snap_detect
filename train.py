import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import models
import time,os,json
#向上取整
UP_INT = lambda x: int(x) if x % 1 == 0 else int(x) + 1

TRAIN_DATASET_PATH="dataset/train"
TEST_DATASET_PATH="dataset/tesst"
NUM_MEL_BINS = 40
EPOCHS = 200
SIMPLE_RATE = 16000
SAMPLE_LENGTH = 7680
BATCH_SIZE = 32
FRAME_LENGTH = 1024
FRAME_STEP = 256
FRAME_NUM = int(UP_INT(SAMPLE_LENGTH - FRAME_LENGTH) / FRAME_STEP + 1) 
print(FRAME_NUM)
LOWER_EDGE_HERTZ = 2000
UPPER_EDGE_HERTZ = 6000
DATE = time.strftime("%m%d_%H%M", time.localtime())
DIR = 'model_%s' % DATE
TF_MODEL = "%s/model_%s.%s"
TFLITE_MODEL = "%s/model_%s.tflite"
ROC_PNG = "%s/roc_%s.png"
MEL_PNG = "%s/mel.png"
SNAP_C = "%s/snap.c"
MODEL_INFO = "%s/model_info_%s.json"
VOICE_TYPE = np.array(tf.io.gfile.listdir(TRAIN_DATASET_PATH))
TYPE_NUM = len(VOICE_TYPE)
if not os.path.exists(DIR):
    os.makedirs(DIR)


class Dataset:
    def __init__(self, dataset_path,train_split=0.9, val_split=0.1):
        self.dataset_path = dataset_path
        train_ds,val_ds = self._get_dataset(TRAIN_DATASET_PATH,0.1)
        try:
            test_ds = self._get_dataset(TEST_DATASET_PATH)
        except:
            test_ds = val_ds.shard(2,1)
        train_ds = train_ds.shuffle(1000)
        val_ds = val_ds.shuffle(1000)
        test_ds = test_ds.shuffle(1000)
        train_ds = train_ds.map(self._squeeze)
        val_ds = val_ds.map(self._squeeze)
        test_ds = test_ds.map(self._squeeze)
        train_ds = self._get_mel(train_ds)
        val_ds = self._get_mel(val_ds)
        test_ds = self._get_mel(test_ds)
        # train_ds = dataset.take(int(len(dataset) * train_split)).cache().shuffle(100).prefetch(tf.data.AUTOTUNE)
        # val_ds = dataset.skip(int(len(dataset) * train_split)).take(int(len(dataset) * val_split)).cache().prefetch(tf.data.AUTOTUNE)
        # test_ds = dataset.skip(int(len(dataset) * (train_split + val_split))).cache().prefetch(tf.data.AUTOTUNE)
        # random_train = train_ds.map(self._random_multiply)
        # random_val = val_ds.map(self._random_multiply)
        # random_test = test_ds.map(self._random_multiply)
        # self.train_ds = train_ds.concatenate(random_train)
        # self.val_ds = val_ds.concatenate(random_val)
        # self.test_ds = test_ds.concatenate(random_test)
        self.train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
        self.val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        self.test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)



    def show(self,save=False):
        lines = BATCH_SIZE ** 0.5 if BATCH_SIZE ** 0.5 % 1 == 0 else BATCH_SIZE ** 0.5 + 1 
        lines = int(lines)
        columns = BATCH_SIZE / lines if BATCH_SIZE / lines % 1 == 0 else BATCH_SIZE / lines + 1
        columns = int(columns)
        for mel, label in self.train_ds.take(1):
            for i in range(BATCH_SIZE):
                plt.subplot(lines , columns, i+1)
                plt.imshow(mel[i].numpy().T, aspect='auto', origin='lower')
                plt.title(VOICE_TYPE[label[i].numpy()])
            if save:
                plt.savefig(MEL_PNG % DIR)
            else:
                plt.show()
    
    def _trim_lowwest(self, audio, label):
        '''
        @param audio: 音频数据
        @param label: 标签数据
        @return audio: 截取最大值为中心随机偏移0.1s，并截断成0.5s的音频数据
        '''
        if (audio.shape[1] < 8000):
            return audio, label
        max_audio = tf.reduce_max(audio,axis=1)
        max_audio_index = tf.argmax(max_audio)
        #随机偏移0.1s
        print(max_audio_index)
        bias = tf.random.uniform(shape=(),minval=0,maxval=0.1,dtype=tf.float32)
        mid = max_audio_index + int(SIMPLE_RATE * bias)
        audio = audio[mid - int(SIMPLE_RATE / 2):mid + int(SIMPLE_RATE / 2)]
        return audio, label        

    def _squeeze(self, audio, label):
        return tf.squeeze(audio,axis=-1), label
    
    def _random_multiply(self, audio, label):
        return audio * tf.random.uniform(shape=(),minval=0.5,maxval=2.5,dtype=tf.float32), label

    def _get_dataset(self,directory,val_split=0.0):
        if val_split:
            return tf.keras.utils.audio_dataset_from_directory(
                directory=directory,
                batch_size=BATCH_SIZE,
                seed=np.random.randint(0,1000),
                output_sequence_length=SAMPLE_LENGTH,
                validation_split=val_split,
                subset="both",
            )
        else:
            return tf.keras.utils.audio_dataset_from_directory(
                directory=directory,
                batch_size=BATCH_SIZE,
                seed=np.random.randint(0,1000),
                output_sequence_length=SAMPLE_LENGTH,
            )
    
    def _get_mel(self, ds):
        return ds.map(
            map_func=lambda audio, label: (Dataset.convert2mel(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    @staticmethod
    def convert2mel(audio, sample_rate=SIMPLE_RATE,num_mel_bins=NUM_MEL_BINS, lower_edge_hertz=LOWER_EDGE_HERTZ, upper_edge_hertz=UPPER_EDGE_HERTZ):
        stfts = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)
        # 获取频谱幅度
        spectrograms = tf.abs(stfts)
        # 计算梅尔权重矩阵
        num_spectrogram_bins = stfts.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
        # 将频谱转换为梅尔频谱
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        print(mel_spectrograms.shape)
        #打印mel_spectrograms里的最大值
        return mel_spectrograms

class Sound_Classification_Model(tf.Module):
    def __init__(self, model_type='CNN'):
        self.model_type = model_type
        if model_type == 'GRU':
            self.model = self.GRU()
        elif model_type == 'CNN':
            self.model = self.CNN()
        elif model_type == 'LSTM':
            self.model = self.LSTM()
        elif model_type == 'DNN':
            self.model = self.DNN()
        else:
            raise ValueError("model_type must be one of ['GRU','CNN','LSTM','DNN']")
        self.summary()
        self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=[None, SAMPLE_LENGTH], dtype=tf.float32))


    @tf.function
    def __call__(self, x):
        if len(x.shape) == 1:
            x = x[tf.newaxis, :]
        x = Dataset.convert2mel(x)
        return self.model(x, training=False)


    def GRU(self):
        return models.Sequential([
            layers.Input(shape=(FRAME_NUM,NUM_MEL_BINS),name='mel'),
            layers.Dropout(0.3),
            layers.GRU(4),
            layers.Dropout(0.3),
            layers.Dense(1,activation='sigmoid')
        ])
    
    def CNN(self):
        model = models.Sequential([
            layers.Input(shape=(FRAME_NUM,NUM_MEL_BINS),name='mel'),
            layers.Reshape((FRAME_NUM, NUM_MEL_BINS, 1)),
            layers.Conv2D(12, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(32,activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1,activation='sigmoid')
        ])
        return model
    
    def LSTM(self):
        model = models.Sequential([
            layers.Input(shape=(FRAME_NUM,NUM_MEL_BINS),name='mel'),
            layers.LSTM(32),
            layers.Dense(1,activation='sigmoid')
        ])
        return model
    
    def DNN(self):
        model = models.Sequential([
            layers.Input(shape=(FRAME_NUM,NUM_MEL_BINS),name='mel'),
            layers.Flatten(),
            layers.Dense(1,activation='sigmoid')
        ])
        return model
    
    def fit(self,train_ds, val_ds, epochs=EPOCHS):
        # total = 1800
        # neg = 1500
        # pos = 300
        # weight_for_0 = (1 / neg)*(total)/2.0 
        # weight_for_1 = (1 / pos)*(total)/2.0

        # class_weight = {0: weight_for_0, 1: weight_for_1}

        # print('Weight for class 0: {:.2f}'.format(weight_for_0))
        # print('Weight for class 1: {:.2f}'.format(weight_for_1))
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            callbacks=[tf.keras.callbacks.TensorBoard(log_dir="./logs")],
            #class_weight=class_weight
            #callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=5),tf.keras.callbacks.TensorBoard(log_dir="./logs")],
        )
    
    def compile(self, optimizer, loss, metrics):
        self.mertics = metrics
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def evaluate(self, test_ds,save=False):
        plt.figure(figsize=(10, 10*3))
        keys = list(self.history.history.keys())
        plt.subplot(3, 1, 1)
        for key in keys:
            if not "loss" in key:
                if not "val" in key:
                    plt.plot(self.history.history[key][20:], label=key)
                else:
                    lable_name = key.replace('val_','')
                    plt.plot(self.history.history[key][20:], label=key)
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(3, 1, 2)
        #从第20个epoch开始绘制
        plt.plot(self.history.history['loss'][20:], label='loss')
        plt.plot(self.history.history['val_loss'][20:], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        test_ds = test_ds.unbatch().batch(1)
        y_true = []
        y_pred = []
        for mel, label in test_ds:
            y_true.append(label.numpy())
            y_pred.append(self.model(mel, training=False).numpy())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(-1)

        # 计算假正率、真正率和阈值
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # 计算AUC值
        auc_value = auc(fpr, tpr)

        # 绘制AUC ROC曲线
        plt.subplot(3, 1, 3)
        plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc_value))
        plt.plot([0, 1], [0, 1], 'k--')  # 对角线
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        # 绘制阈值点
        for threshold in thresholds:
            idx = np.where(thresholds == threshold)[0][0]
            plt.scatter(fpr[idx], tpr[idx], c='red', label='Threshold = {:.2f}'.format(threshold))
            plt.text(fpr[idx], tpr[idx], '{:.2f}'.format(threshold), verticalalignment='bottom', horizontalalignment='right')
        #找到fpr为0.01的阈值
        idx = np.where(fpr < 0.01)[0][-1]
        self.BEST_THRESHOLD = thresholds[idx]
        #转换为python float
        self.BEST_THRESHOLD = float(self.BEST_THRESHOLD)
        if save:
            plt.savefig(ROC_PNG % (DIR,self.model_type))
        else:
            plt.show()
        self.EVALUATE = self.model.evaluate(test_ds, return_dict=True)
    
    def save(self, filepath, save_format='h5'):
        if save_format == 'h5':
            filepath = filepath % (DIR,self.model_type,save_format)
            self.model.save(filepath=filepath, save_format=save_format)
        elif save_format == 'export':
            filepath = filepath % (DIR,self.model_type,save_format)
            tf.saved_model.save(self, filepath)

        MODEL_JSON = {
    "train_dataset_path" : TRAIN_DATASET_PATH,
    "model_type" : self.model_type,
    "roc" : ROC_PNG % (DIR,self.model_type),
    "mel" : MEL_PNG % DIR,
    "snap" : SNAP_C % DIR,
    "model" : TF_MODEL % (DIR,self.model_type,save_format),
    "tflite" : TFLITE_MODEL % (DIR,self.model_type),
    "frame_num" : FRAME_NUM,
    "frame_length" : FRAME_LENGTH,
    "frame_step" : FRAME_STEP,
    "lower_edge_hertz" : LOWER_EDGE_HERTZ,
    "upper_edge_hertz" : UPPER_EDGE_HERTZ,
    "num_mel_bins" : NUM_MEL_BINS,
    "best_threshold" : self.BEST_THRESHOLD,
    "EVALUATE" : self.EVALUATE,
    "date" : DATE
}
        with open(MODEL_INFO % (DIR,self.model_type), 'w') as f:
            json.dump(MODEL_JSON, f)

    def summary(self):
        return self.model.summary()

dataset = Dataset(TRAIN_DATASET_PATH)
train, val, test = dataset.train_ds, dataset.val_ds, dataset.test_ds
dataset.show(save=True)
initial_learning_rate = 0.001
decay_steps = int(EPOCHS/10)
decay_rate = 1/10
lr_schedule =  tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate)
CNN = Sound_Classification_Model('CNN')
CNN.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['AUC'],
)
CNN.fit(train, val, epochs=EPOCHS)
CNN.evaluate(test,save=True)
CNN.save(TF_MODEL , save_format='h5')
TFLITE_OUTPUT = TFLITE_MODEL % (DIR,'CNN')
TP = 0
FP = 0
TN = 0
FN = 0
# def representative_data_gen():
#   for input_value,lable in train.unbatch().batch(1).take(100):
#     # Model has only one input so each data point has one element.
#     yield [input_value]
converter = tf.lite.TFLiteConverter.from_keras_model(CNN.model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter._experimental_lower_tensor_list_ops = False
#tflite_model=converter.convert()
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_data_gen
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
open(TFLITE_OUTPUT, "wb").write(tflite_model)
interpreter = tf.lite.Interpreter(model_path=TFLITE_OUTPUT)
interpreter.allocate_tensors() 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

flag = 0
for mel, label in test.unbatch().batch(1):
    #mel = mel[...,tf.newaxis]
    interpreter.set_tensor(input_details[0]['index'], mel)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #打印TP
    if label == 1 and output_data[0][0] > 0.90:
        TP += 1
        #将第一个遇到的转为c数组，保存在snap.c
        if flag == 0:
            flag = 1
            mel_c_array = np.array(mel).flatten()
            mel_c_array = mel_c_array.astype(np.float32)
            file = open(os.path.join(DIR,"snap.c"),"w")
            line = []
            line.append("const float mel[] = {")
            for i in range(len(mel_c_array)):
                line.append(str(mel_c_array[i]))
                line.append(",")
            line.append("};")
            file.write("".join(line))
            file.close()
    if label == 0 and output_data[0][0] > 0.90:
        FP += 1
    if label == 0 and output_data[0][0] < 0.90:
        TN += 1
    if label == 1 and output_data[0][0] < 0.90:
        FN += 1
    #print(label, output_data[0][0])
print("TP:", TP)
print("FP:", FP)
print("TN:", TN)
print("FN:", FN)
os.system("xxd -i %s > %s" % (TFLITE_OUTPUT, TFLITE_OUTPUT + ".h"))
