import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras
import argparse
import matplotlib.pyplot as plt

# Echo tensorflow version
print(tf.__version__)

# # Just disables the warning, doesn't take advantage of AVX/FMA to run faster
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Command line arguments
parser = argparse.ArgumentParser(prog='helloworld', allow_abbrev=True)
parser.add_argument("--epoch", help="Define epoch", type=int, default=30)
parser.add_argument("--loss", help="Acceptable loss", type=float, default=0.4)
args = parser.parse_args()

# Set values
EPOCH=args.epoch
LOSS=args.loss

# Load data set
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Show 1 sample
np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

# Normalzie data
training_images  = training_images / 255.0
test_images = test_images / 255.0


# Net
# Input: 255 Inputs commign from the image dimensions 28x28 converted to linear
# Hidden layer: will do the calculation to transform the 128 into the 10 categories
# Ouput: 10 different categoties of images

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


# Compile 
# - using Stocastich Gradient Decent optimizer
# - mean^2 error
# - optimizer: next guess shoudl be better that the one before
# - loss: calculate how good or bad the guess
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
              

# Train $EPOCH times loop

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<LOSS):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
model.fit(training_images, training_labels, epochs=EPOCH, callbacks=[callbacks])

# Predict
print(model.predict([PREDICT]))


#
# REFERENCE
#

# Throubleshooting precision issue insode container
# "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA"
#
# - https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u

# Command line arguments
#
# - https://www.onlinetutorialspoint.com/python/how-to-pass-command-line-arguments-in-python.html
# - https://docs.python.org/3/library/argparse.html
# - https://www.golinuxcloud.com/python-argparse/
