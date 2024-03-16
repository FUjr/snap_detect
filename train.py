import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import models
import time,os,json

TRAIN_DATASET_PATH="dataset/train"
NUM_MEL_BINS = 15
EPOCHS = 100
SIMPLE_RATE = 16000
SECOND = 1
BATCH_SIZE =32
FRAME_LENGTH = 2000
FRAME_STEP = 1000
FRAME_NUM = int(16000 / FRAME_STEP - (FRAME_LENGTH / FRAME_STEP - 1))
LOWER_EDGE_HERTZ = 1500
UPPER_EDGE_HERTZ = 5500
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
    def __init__(self, dataset_path,train_split=0.8, val_split=0.1, test_split=0.1):
        self.dataset_path = dataset_path
        dataset = self._get_dataset()
        if train_split + val_split + test_split != 1:
            raise ValueError("train_split + val_split + test_split must be 1")
        dataset = dataset.shuffle(100)
        dataset = dataset.map(self._squeeze)
        dataset = self._get_mel(dataset)
        self.train_ds = dataset.take(int(len(dataset) * train_split)).cache().shuffle(100).prefetch(tf.data.AUTOTUNE)
        self.val_ds = dataset.skip(int(len(dataset) * train_split)).take(int(len(dataset) * val_split)).cache().prefetch(tf.data.AUTOTUNE)
        self.test_ds = dataset.skip(int(len(dataset) * (train_split + val_split))).cache().prefetch(tf.data.AUTOTUNE)

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
        
    def _squeeze(self, audio, label):
        return tf.squeeze(audio,axis=-1), label
    
    def _get_dataset(self):
        return tf.keras.utils.audio_dataset_from_directory(
            directory=self.dataset_path,
            batch_size=BATCH_SIZE,
            seed=np.random.randint(0,1000),
            output_sequence_length=SIMPLE_RATE * SECOND,
        )
    
    def _get_mel(self, ds):
        return ds.map(
            map_func=lambda audio, label: (Dataset.convert2mel(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    @staticmethod
    def convert2mel(audio, sample_rate=SIMPLE_RATE,num_mel_bins=NUM_MEL_BINS, lower_edge_hertz=LOWER_EDGE_HERTZ, upper_edge_hertz=UPPER_EDGE_HERTZ):
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
        return mel_spectrograms

class Sound_Classification_Model(tf.Module):
    def __init__(self, model_type='GRU'):
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
        x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))


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
            layers.Conv2D(3, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
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
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=5),
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
                    plt.plot(self.history.history[key], label=key)
                else:
                    lable_name = key.replace('val_','')
                    plt.plot(self.history.history[key], label=lable_name)
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
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
CNN = Sound_Classification_Model('GRU')
CNN.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['AUC','accuracy'],
)
CNN.fit(train, val, epochs=EPOCHS)
CNN.evaluate(test,save=True)
CNN.save(TF_MODEL , save_format='export')
