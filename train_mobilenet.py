import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
import tensorflow_hub as hub
import numpy as np
import os 


#model params
HEIGHT = 224
WIDTH = 224
DEPTH = 3 
IMAGE_SHAPE = (HEIGHT, WIDTH)
num_classes = 491
BATCH_SIZE = 10
INPUT_TENSOR_NAME = "inputs_input" # According to Amazon, needs to match the name of the first layer + "_input"

#a sagemaker entry script requires 4 functions keras_model_fn, servinf_input_fn, train_input_fn, eval_input_fn

#defines and compiles the model 
def keras_model_fn(hyperparameters):
  # Google's feature extractor
  feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

  # Downscales image size to 224x224(x3)
  
  # Getting MobileNet features
  model = tf.keras.Sequential([
      hub.KerasLayer(feature_extractor_url, output_shape=[1280]),  # Can be True, see below.
      tf.keras.layers.Dense(num_classes, activation='softmax')
  ])

  model.build([None,HEIGHT, WIDTH, DEPTH])

  # compile the model
  model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['acc'])
  model.summary()
  return model

#this function describes how data is fed to the model. copied as is from https://blog.betomorrow.com/keras-in-the-cloud-with-amazon-sagemaker-67cf11fb536
def serving_input_fn(hyperparameters):
    tensor = tf.placeholder(tf.float32, shape=[None,224,224,3])
    inputs = {INPUT_TENSOR_NAME: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

#returns training images and their correspnding labls
def train_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE, data_dir=training_dir)

#returns evaluation images and their correspnding labls
def eval_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE, data_dir=training_dir)

#
def _input(mode, batch_size, data_dir):
  assert os.path.exists(data_dir), ("Unable to find images resources for input, are you sure you downloaded them ?")
  #if called by train input
  if mode == tf.estimator.ModeKeys.TRAIN:
    #do image augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.,
                                                                  horizontal_flip=True,
                                                                  rotation_range=40,
                                                                  width_shift_range=0.2,
                                                                  height_shift_range=0.2,
                                                                  shear_range=0.2,
                                                                  zoom_range=0.2,
                                                                  fill_mode='nearest')
  #else if training. just rescale dont do augmentation
  else:
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
  
  generator = datagen.flow_from_directory(data_dir, target_size=(HEIGHT, WIDTH), class_mode='categorical', batch_size=batch_size)
  images, labels = generator.next()

  return {INPUT_TENSOR_NAME: images}, labels




