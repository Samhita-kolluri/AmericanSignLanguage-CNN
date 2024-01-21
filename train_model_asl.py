import os
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

for dirname, _, filenames in os.walk('MNSIT/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df = pd.read_csv("MNSIT/sign_mnist_train/sign_mnist_train.csv")
validate_df = pd.read_csv("MNSIT/sign_mnist_validate/sign_mnist_validate.csv")

validate = pd.read_csv("MNSIT/sign_mnist_validate/sign_mnist_validate.csv")
y = validate['label']

# plt.figure(figsize=(5, 5))
# sns.countplot(train_df['label'])

y_train = train_df['label']
y_validate = validate_df['label']
del train_df['label']
del validate_df['label']

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_validate = label_binarizer.fit_transform(y_validate)

x_train = train_df.values
x_validate = validate_df.values

x_train = x_train / 255
x_validate = x_validate / 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_validate = x_validate.reshape(-1, 28, 28, 1)

# f, ax = plt.subplots(2, 5)
# f.set_size_inches(10, 10)
# k = 0
# for i in range(2):
#     for j in range(5):
#         ax[i, j].imshow(x_train[k].reshape(28, 28), cmap="gray")
#         k += 1
#     plt.tight_layout()

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=24, activation='softmax'))
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(datagen.flow(np.array(x_train), np.array(y_train), batch_size=128), epochs=60,
                 validation_data=(np.array(x_validate), np.array(y_validate)))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

model.save('Models/base_model_asl_60')
