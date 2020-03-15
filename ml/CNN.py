
'''

Hand Pose Convolutional Neural Network

The Keras implementation based on the CPM CNN denoted here:
https://github.com/timctho/convolutional-pose-machines-tensorflow/blob/master/models/nets/cpm_hand.py

'''

# Libraries required for model construction
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential

# Helper libraries
import numpy as np

class Hand_CNN(keras.Model):

    # Assumes all images are color (RGB)
    def __init__ (self, input_size, heatmap_size, stages, joints, is_training=True):
        self.stages = stages
        self.input_size = input_size
        self.stage_heatmap = []
        self.stage_loss = [0 for _ in range(stages)]
        self.total_loss = 0
        self.input_image = None
        self.center_map = None
        self.gt_heatmap = None
        self.init_lr = 0
        self.merged_summary = None
        self.joints = joints
        self.batch_size = 0
        self.inference_type = 'Train'
        self.input_images = tf.keras.backend.placeholder(dtype=tf.float32,
                                               shape=(None, input_size, input_size, 3),
                                               name='input_placeholder')

        self.cmap_placeholder = tf.keras.backend.placeholder(dtype=tf.float32,
                                               shape=(None, input_size, input_size, 1),
                                               name='cmap_placeholder')
        self.gt_hmap_placeholder = tf.keras.backend.placeholder(dtype=tf.float32,
                                                  shape=(None, heatmap_size, heatmap_size, joints + 1),
                                                  name='gt_hmap_placeholder')

        self.build_model()

    # Call function for a custom forward pass (from keras.Model)
    def call(self, input):
        layer1_output = self.layer1(input)
        return self.layer2(layer1_output)

    # This function initializes a new Hand Gesture CNN based on the constructor arguments
    def build_model(self):

        # Begin model layering construction procedure
        model = Sequential()
        
        model.add(keras.layers.Conv2D(
            input_shape = (self.input_size, self.input_size, 3),
            filters = 64,
            kernel_size = [3,3],
            strides = [1, 1],
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.glorot_normal
        ))

        model.add(keras.layers.Conv2D(
            filters = 64,
            kernel_size = [3, 3],
            strides = [1, 1],
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.glorot_normal
        ))

        model.add(keras.layers.MaxPool2D(
            pool_size = [2,2],
            strides = 2
        ))

        model.add(keras.layers.Conv2D(
            filters = 128,
            kernel_size = [3, 3],
            strides = [1, 1],
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.glorot_normal
        ))

        model.add(keras.layers.Conv2D(
            filters = 128,
            kernel_size = [3, 3],
            strides = [1, 1],
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.glorot_normal
        ))

        model.add(keras.layers.MaxPool2D(
            pool_size = [2,2],
            strides = 2
        ))

        model.add(keras.layers.Conv2D(
            filters = 256,
            kernel_size = [3, 3],
            strides = [1, 1],
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.glorot_normal
        ))

        model.add(keras.layers.Conv2D(
            filters = 128,
            kernel_size = [3, 3],
            strides = [1, 1],
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.glorot_normal
        ))

        model.add(keras.layers.MaxPool2D(
            pool_size = [2,2],
            strides = 2
        ))

        model.add(keras.layers.Conv2D(
            filters = 512,
            kernel_size = [3, 3],
            strides = [1, 1],
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.glorot_normal
        ))

        model.add(keras.layers.Conv2D(
            filters = 512,
            kernel_size = [3, 3],
            strides = [1, 1],
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.glorot_normal
        ))

        model.add(keras.layers.Conv2D(
            filters = 512,
            kernel_size = [3, 3],
            strides = [1, 1],
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.glorot_normal
        ))

        model.add(keras.layers.Conv2D(
            filters = 512,
            kernel_size = [3, 3],
            strides = [1, 1],
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.glorot_normal
        ))

        model.add(keras.layers.Conv2D(
            filters = 128,
            kernel_size = [3, 3],
            strides = [1, 1],
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.glorot_normal
        ))

        self.model = model
        
    def get_model(self):
        return self.model

    def load_model_weights(self, config):
        self.model = Sequential.from_config(config)