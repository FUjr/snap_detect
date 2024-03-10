import tensorflow as tf

load_model = tf.saved_model.load('voice_model')
file = tf.data.Dataset.list_files('.\\*.wav', shuffle=False)
for f in file:
  result = load_model(f)
  print(f)
  print(result['class_names'].numpy()[0].decode('utf-8'))
  print(result['predictions'].numpy())
  print(result['class_ids'].numpy())
