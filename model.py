import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D,  Cropping2D
from keras.layers import MaxPooling2D, regularizers, Lambda#, BatchNormalization, optimizers
import matplotlib.pyplot as plt
import cv2
import sklearn
from sklearn.utils import shuffle
import csv
import random

# Merge all camera images ie cnter, left and right lane with correction of 1.8 and creates new csv file as driving.csv this will be easy to analyze.
with open(r"C:\Users\pramo\Documents\Project3\driving.csv", "w", newline='') as outfile:
    writer = csv.writer(outfile, delimiter=",")
    with open (r"C:\Users\pramo\Documents\Project3\driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)       
        for line in reader:   
             for i in range(3):
                 correction = 0.18
                 if i == 1:
                    angle = (float(line[3]) + correction)
                 elif i == 2:
                    angle = (float(line[3]) - correction)
                 else:
                    angle = float(line[3])
                    
                 #angle = round(angle, 2)
                 writer.writerow([line[i], angle])

# Read the new csv file.                 
samples = []
with open (r"C:\Users\pramo\Documents\Project3\driving.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Random suffle 
random.shuffle(samples)      

# Used sklearn train_test_split to split test and traning, with validation size of 0.2* total size(21000)     
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2, shuffle=True)

# Batch Size
Size = 32

# Generator.
def generator(samples, batch_size=Size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
                
            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                image = cv2.imread(source_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                angle = float(batch_sample[1])
                angles.append(angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# This Section generates the traning data as decribe in 18th Chapter.
train_generator = generator(train_samples, batch_size=Size)
# This Section generates the validation data as decribe in 18th Chapter.
validation_generator = generator(validation_samples, batch_size=Size)

# Model 
model = Sequential()
# Normilize input
model.add(Lambda(lambda x:(x / 255.0) - 0.5, input_shape=(160,320,3)))
# Cropping image top 70 rows and bottom 25 rows
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Filter size 3x3 depth 8 and nonlinearity (activation) relu. 
model.add(Conv2D(8,(3,3), activation='elu'))
#model.add(BatchNormalization())
# Added Max pooling layer 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Filter size 3x3 depth 16 and nonlinearity (activation) relu. 
model.add(Conv2D(16,(3,3), activation='elu'))
#model.add(BatchNormalization())
# Added Max pooling layer 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Filter size 3x3 depth 32 and nonlinearity (activation) relu. 
model.add(Conv2D(32,(3,3), activation='elu'))
#model.add(BatchNormalization())
# Added Max pooling layer 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Added Flatten layer 
model.add(Flatten())
# Added Dense layer with output 100, with activation relu and regularizers.l2 to overcome the overfitting 
model.add(Dense(100, activation='elu', kernel_regularizer=regularizers.l2(0.003)))
#model.add(BatchNormalization())
# Added Dense layer with output 50, with activation relu and regularizers.l2 to overcome the overfitting 
model.add(Dense(50, activation='elu', kernel_regularizer=regularizers.l2(0.003)))
#model.add(BatchNormalization())
# Added Dense layer with output 10, with activation relu and regularizers.l2 to overcome the overfitting 
model.add(Dense(10, activation='elu', kernel_regularizer=regularizers.l2(0.003)))
#model.add(BatchNormalization())
# Added Dense layer with output 1 that will be steering angle
model.add(Dense(1))
# Used ADAM optimizer with learning rate 0.00001
#ADAM = optimizers.Adam(lr=0.00001)
model.compile(loss='mse', optimizer='Adam')
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)

# Print output
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
# Saves the Model. 
model.save("model.h5")