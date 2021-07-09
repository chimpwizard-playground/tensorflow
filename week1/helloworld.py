import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras
import argparse

# # Just disables the warning, doesn't take advantage of AVX/FMA to run faster
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Command line arguments
parser = argparse.ArgumentParser(prog='helloworld', allow_abbrev=True)
parser.add_argument("--epoch", help="Define epoch", type=int, default=500)
parser.add_argument("--predict", help="NUmber to predict", type=float, default=10.0)
args = parser.parse_args()

# Set values
EPOCH=args.epoch
PREDICT=args.predict


# 1 Layer 1 Neuron, 1 Input
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile 
# - using Stocastich Gradient Decent optimizer
# - mean^2 error
# - optimizer: next guess shoudl be better that the one before
# - loss: calculate how good or bad the guess
model.compile(optimizer='sgd', loss='mean_squared_error')

# Known data
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train EPOCH times loop
model.fit(xs,ys, epochs=EPOCH)

#Evaluate using the unseen data
model.evaluate(test_images, test_labels)

# Predict
classifications = model.predict(test_images)
print(classifications[0])


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
