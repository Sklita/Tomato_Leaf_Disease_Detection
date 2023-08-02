import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os
import warnings

warnings.filterwarnings('ignore')

excel_file_path = 'C:/TLDExcel/Data.xlsx'
excel_table = pd.read_excel(excel_file_path)

model_path = 'C:/TLDExcel/model_weights_new.h5'

number_images = excel_table.shape[0]
disease_list = ['Bacterial Spot', 'Early Blight', 'Healthy',
                'Late Blight', 'Leaf Mold', 'Mosaic Virus',
                'Septoria Leaf Spot', 'Spider Mites Two Spotted Spider Mite',
                'Target Spot', 'Yellow Leaf Curl Virus']

cnn = keras.models.load_model(model_path)
print('Running Analysis...')
for i in range(0, number_images):
    temp_img_path = excel_table['Image Path'][i]
    temp_img_path.replace('\\', '/')
    
    with tf.device('/cpu:0'):
        test_img = image.load_img(temp_img_path, target_size = (128, 128))
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis = 0)
        result = cnn.predict(test_img / 255.0)

    final_list = []
    for j in range(0, 10):
        temp_tuple = tuple((disease_list[j], result[0][j]))
        final_list.append(temp_tuple)

    final_list.sort(key = lambda x:x[1])
    excel_table['Result'][i] = final_list[9][0]

excel_table.to_excel(excel_file_path, index = False)
print('Completed') 
