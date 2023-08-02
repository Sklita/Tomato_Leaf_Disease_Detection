import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os

cwd = os.getcwd()
cwd.replace('\\', '/')

read_file_path = cwd + "/commPR.txt"
write_file_path = cwd + "/commPW.txt"
fr = open(read_file_path, 'r')
filePath = str(fr.read().splitlines()[0])
filePath.replace("\\", "/")
fr.close()
open(write_file_path, "w").close()
print("Running Analysis...")
fw = open(write_file_path, 'r+')

with tf.device('/cpu:0'):
    cnn = keras.models.load_model(cwd + '/model_weights_final.h5')

    test_image = image.load_img(filePath, target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image / 255.0)

dir_list = ['Bacterial Spot', 'Early Blight', 'Healthy']

temp = []
for i in range(len(dir_list)):
    temp.append(result[0][i] * 100)


for i in range(len(dir_list)):
    fw.write(str(dir_list[i]) + ": " + str(temp[i]) + "\n")

fw.close()






