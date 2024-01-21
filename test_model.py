from PIL import Image
import numpy as np
import pandas as pd
import os
import csv
import keras
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from operator import itemgetter

model = load_model('Models/base_model_asl_50')

image_file = 'test_img.jpg'
img = Image.open(image_file)

# width, height = img.size
# format = img.format
# mode = img.mode

img_grey = img.convert('L')

input_size = (28, 28)
img_csv = img_grey.resize(input_size)

value = np.asarray(img_csv.getdata(), dtype=np.int).reshape((img_csv.size[1], img_csv.size[0]))
value = value.flatten()
# header = list()

# for i in range(1, 785):
#     title = "pixel"
#     num = str(i)
#     title = title + num
#     header.append(title)

# with open("img_pixels.csv", 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
#     writer.writerow(value)

value_norm = value / 255
input_value = value_norm.reshape(-1, 28, 28, 1)

output_value = model.predict_classes(input_value)

result = "ERROR"

if output_value[0] >= 9:
    output_value[0] += 1

if output_value == 0:
    result = "A"

elif output_value == 1:
    result = "B"

elif output_value == 2:
    result = "C"

elif output_value == 3:
    result = "D"

elif output_value == 4:
    result = "E"

elif output_value == 5:
    result = "F"

elif output_value == 6:
    result = "G"

elif output_value == 7:
    result = "H"

elif output_value == 8:
    result = "I"

elif output_value == 10:
    result = "K"

elif output_value == 11:
    result = "L"

elif output_value == 12:
    result = "M"

elif output_value == 13:
    result = "N"

elif output_value == 14:
    result = "O"

elif output_value == 15:
    result = "P"

elif output_value == 16:
    result = "Q"

elif output_value == 17:
    result = "R"

elif output_value == 18:
    result = "S"

elif output_value == 19:
    result = "T"

elif output_value == 20:
    result = "U"

elif output_value == 21:
    result = "V"

elif output_value == 22:
    result = "W"

elif output_value == 23:
    result = "X"

elif output_value == 24:
    result = "Y"

else:
    result = "Z"


print(result)

# print(max(enumerate(output_value[0]), key=itemgetter(1))[0])
