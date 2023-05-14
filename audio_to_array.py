import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Data path
DATASET_PATH = 'data'
data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
  print("Error in data path")

# Key Words
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

# Check audio files
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])

# Read audio files
test_file = tf.io.read_file(DATASET_PATH+'/hello/WhatsApp-Ptt-2023-05-14-at-21.17.16.wav')
test_audio, _ = tf.audio.decode_wav(contents=test_file)
print("Shape: ", test_audio.shape)
print("test_audio: ", test_audio)

plt.plot(test_audio)
plt.show()