import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from tensorflow.keras import layers
from tensorflow.keras import models

TRAIN_DATASET_PATH="dataset\\train"
TEST_DATASET_PATH="dataset\\test"
voice_type = np.array(tf.io.gfile.listdir(TRAIN_DATASET_PATH))
type_num = len(voice_type)

train_ds ,val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=TRAIN_DATASET_PATH,
    batch_size=16,
    seed=123,
    output_sequence_length=16000,
    validation_split=0.1,
    subset='both',
)
test_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=TEST_DATASET_PATH,
    batch_size=16,
    seed=123,
    output_sequence_length=16000,
)

def squeeze(audio, label):
    return tf.squeeze(audio,axis=-1), label



def get_spectrogram(audio):
    #转换为频谱图
    spectrogram = tf.signal.stft(
      audio, frame_length=512, frame_step=64)
    #转换为幅度谱
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # YOu could add additional signatures for a single wave, or a ragged-batch. 
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))


  @tf.function
  def __call__(self, x):
    # If they pass a string, load the file and decode it. 
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]
    x = get_spectrogram(x)  
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return {'predictions':result,
            'class_ids': class_ids,
            'class_names': class_names}

#如果是双声道，转换为单声道
label_names = np.array(train_ds.class_names)
train_ds = train_ds.map(squeeze)

val_ds = val_ds.map(squeeze)
test_ds = test_ds.map(squeeze)
train_spec_ds = make_spec_ds(train_ds)
val_spec_ds = make_spec_ds(val_ds)
test_spec_ds = make_spec_ds(test_ds)

for example_spectrograms, example_labels in train_spec_ds.take(1):
    
    break

train_spec_ds = train_spec_ds.cache().shuffle(100).prefetch(tf.data.AUTOTUNE)
val_spec_ds = val_spec_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spec_ds = test_spec_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)


# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spec_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(128, 128),
    # Normalize.
    norm_layer,
    layers.Conv2D(128, 3, activation='relu'),
    layers.Conv2D(256, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(type_num),
])



model.summary()
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
EPOCHS = 100
history = model.fit(
    train_spec_ds,
    validation_data=val_spec_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=5),
)
metrics = history.history
test_eva = model.evaluate(test_spec_ds, return_dict=True)



plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss', 'test_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')
plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.show() 
model.save(filepath='voice_classfication.h5')

model = tf.keras.models.load_model('voice_classfication.h5')
export = ExportModel(model)
tf.saved_model.save(export, 'voice_model')
