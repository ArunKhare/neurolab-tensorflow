import tensorflow as tf
import os
from tensorflow.keras import layers, models
import tensorflow

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("TensorFlow version:", tf.__version__)

# load the mnist dataaset

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the input images

x_train, x_test = x_train / 255.0, x_test / 255.0

# define a function to create a CNN model
def create_model():
  model = models.Sequential()
  model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(28,28,1)))
  model.add(layers.MaxPooling2D(2,2))
  model.add(layers.Conv2D(64,(3,3),activation="relu"))
  model.add(layers.MaxPooling2D(2,2))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10,activation='softmax'))
  return model

# Model1: CNN with fewer parameters
model1 = create_model()
model1.summary()

#Compile and train the model
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(x_train,y_train, epochs=10, validation_data=(x_test,y_test))

# Model 2: CNN with fewer parameters and a different architecture
model2 = models.Sequential()
model2.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(10, activation='softmax'))
model2.summary()

# Compile and train the model
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Model 3: CNN with fewer parameters and a different architecture
model3 = models.Sequential()
model3.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Flatten())
model3.add(layers.Dense(10, activation='softmax'))
model3.summary()

# Compile and train the model
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model3.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

