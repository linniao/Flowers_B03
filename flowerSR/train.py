import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import wandb
from wandb.keras import WandbCallback

# run = wandb.init()
run = wandb.init(project='superres')        #返回一个运行对象，可以通过导入wandb和调用来访问代码中任何位置的运行对象 此运行所属的项目名称
config = run.config                         #

config.num_epochs = 50
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

#两种操作的epoch             训练和测试
config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size


def image_generator(batch_size, img_dir):
    input_filenames = glob.glob(img_dir + "/*-in.jpg")                              #读入图片
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        random.shuffle(input_filenames)                                             #打乱顺序
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (small_images, large_images)                                          #返回一个batch_size的两组图片
        counter += batch_size                                                       #处理下一个batch_size的图片

val_generator = image_generator(config.batch_size, val_dir)     #传入test目录和batch_size 对图片进行分批处理
in_sample_images, out_sample_images = next(val_generator)

#记录图像
class ImageLogger(Callback):                                                        #回调函数
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)


model = Sequential()
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same',
                        input_shape=(config.input_width, config.input_height, 3)))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mse')

#损失函数为均方误差

model.fit_generator(image_generator(config.batch_size, train_dir),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, 
                    callbacks=[ImageLogger(), WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=val_generator)
#callbacks 回调使用这两个函数
#validation_steps 停止钱生成的总步数
#Wandb自动保存所有指标和所跟踪的损失值