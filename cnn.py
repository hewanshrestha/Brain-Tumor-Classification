# Image Classification

# Import libraries
import matplotlib
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Initalize CNN
classifier = Sequential()
epochs = 40
# Add 2 convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add 2 more convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add 2 more convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add global average pooling layer
classifier.add(GlobalAveragePooling2D())

# Add full connection
classifier.add(Dense(units=2, activation='softmax'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        '/media/anondel/Documents/Program/braintumor/train/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        '/media/anondel/Documents/Program/braintumor/valid/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

history = classifier.fit_generator(
        train_set,
        steps_per_epoch=80,
        epochs=epochs,
        validation_data=test_set,
        validation_steps=20)

classifier.save('model.h5')

plt.hist(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend(['train','val'])
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend(['train','val'])
plt.show()
