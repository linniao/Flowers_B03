import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import wandb

batch_size = 32
input_height = 32
input_width = 32
output_height = 256
output_width = 256

# def load_trained_model_v1():
#     root_dir = "./wandb/model_1/"
#     # network_architecture and network_weights are seperated stored
#
#     # json_path = root_dir + 'wandb-summary.json'
#     # load_file = open(json_path, 'r', encoding='utf-8')
#     # json_string = json.loads(load_file.read())
#     # model = model_from_json(str(json_string))
#
#     model = Sequential()
#     model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same',
#                             input_shape=(input_width, input_height, 3)))
#     model.add(layers.UpSampling2D())
#     model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
#     model.add(layers.UpSampling2D())
#     model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
#     model.add(layers.UpSampling2D())
#     model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
#
#     model.load_weights(root_dir + "model-best.h5")
#     return model
    
# def load_trained_model_v2():
#     root_dir = "./wandb/model_2/"
#
#     model = Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
#                             input_shape=(input_width, input_height, 3)))
#     model.add(layers.UpSampling2D())
#     model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
#     model.add(layers.UpSampling2D())
#     model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
#     model.add(layers.UpSampling2D())
#     model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
#
#     model.load_weights(root_dir + "model-best.h5")
#     return model

def load_trained_model_v3():
    root_dir = "./wandb/model_3/"

    model = Sequential()
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same',
                            input_shape=(input_width, input_height, 3)))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))

    model.load_weights(root_dir + "model-best.h5")
    return model


def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")          #获取该文件夹中的所有文件
    counter = 0
    while True:                                                 #死循环
        small_images = np.zeros(                                #输入图片 32*32
            (batch_size, input_width, input_height, 3))
        large_images = np.zeros(                                #输出图片 256*256
            (batch_size, output_width, output_height, 3))
        random.shuffle(input_filenames)                         #将input_filenames中的元素打乱
        if counter+batch_size >= len(input_filenames):          #余下的图片不够一个batchsize的时候 counter=0 开始新一轮epoch
            counter = 0
        for i in range(batch_size):                             #一个批次为32
            img = input_filenames[counter + i]                  #将该文件夹中的图片提取出来
            small_images[i] = np.array(Image.open(img)) / 255.0 #将该文件的图片内容以数组的方式读出来
            large_images[i] = np.array(Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0 #把-in.jpg替换为-out.jpg
        yield (small_images, large_images)                      #返回两个值 small_images 和 large_images
        counter += batch_size                                   #去下一个batch_size

save_path = "./my_test/test_results_v3/"
if not os.path.exists(save_path):                               #没有该目录则创建该目录
    os.mkdir(save_path)

model = load_trained_model_v3()                                 #采用第一个模型
val_dir = 'data/test'                                           #测试集
val_generator = image_generator(batch_size, val_dir)
epoch_num = len(glob.glob(val_dir + "/*-in.jpg")) // batch_size #通过取整来确定具体有多少个epoch
for index in range(epoch_num):
    if index == 2: break
    in_sample_images, out_sample_images = next(val_generator)   #获取经过处理的smallimage和largeimage
    preds = model.predict(in_sample_images)
    #输入测试数据，输出预测的结果
    in_resized = []
    for arr in in_sample_images:
        # Simple upsampling
        in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))  #将数组arr中的行和列的各数值的数量变为原来的8倍，加到in_resized数组的后面
    for i, o in enumerate(preds):                                   #将数组下标和数据分别传给i和o
        img = np.array([o * 255])                                   #得到原图像
        img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy array --> PIL Image
        img_name = save_path + 'example_' + str(int(index*batch_size+i)) + '.jpg'
        img.save(img_name, 'jpeg')
    print("batch", index+1, "has been tested.")
