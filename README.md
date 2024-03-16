# 基于深度学习的响指识别

## 部署

### 1. 准备工作

   你需要有一个麦克风

### 2. 克隆本项目

   ```shell
   git clone git@github.com:fujr/snap_detect
   ```

### 3. 准备python环境

   ```shell
   python -m venv venv
   source venv/bin/activate 
   venv\Scripts\activate (Windows)
   pip install -r requirements.txt
   ```

### 4. 启动

   1. 训练（非必须）：使用record.py录制环境音和你的响指声音，录制完的文件默认在saved_audio目录下，你需要移动到dataset/train下

      ```
      python record.py
      cp -r ./saved_audio/* ./dataset/train/
      python train.py
      
      ```

      训练时，这些参数是可以改动的

      ```python
      #train.py
      #dataset参数和训练效果导出的文件名
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
      
      
      #训练过程
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
      CNN.save(TF_MODEL , save_format='h5')
      LSTM = Sound_Classification_Model('LSTM')
      LSTM.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
          loss=tf.keras.losses.BinaryCrossentropy(),
          metrics=['accuracy'],
      )
      LSTM.fit(train, val, epochs=EPOCHS)
      LSTM.evaluate(test,save=True)
      LSTM.save(TF_MODEL , save_format='keras')
      GRU = Sound_Classification_Model('GRU')
      GRU.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
          loss=tf.keras.losses.BinaryCrossentropy(),
          metrics=['AUC'],
      )
      GRU.fit(train, val, epochs=EPOCHS)
      GRU.evaluate(test,save=True)
      GRU.save(TF_MODEL , save_format='export')
      ```

      

   2. 推理 使用模型进行推理

      ```shell
      python infer.py 模型训练时间 网络类型 模型格式
      #如:3月17日 01：02分训练的LSTM网络 h5格式模型的启动方法为
      python infer.py 0316_0102 LSTM h5
      ```

   
