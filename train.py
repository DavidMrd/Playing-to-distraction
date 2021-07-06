import pandas as pd
import numpy as np
#from google.colab import drive


# %pip install tf-explain tensorflow==2.2.0
from tensorflow.keras.preprocessing import image
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.grad_cam import GradCAM
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib.pyplot as plt
import cProfile

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D,GlobalAveragePooling2D, Dense , BatchNormalization, Dropout, Conv2D, MaxPooling2D,Activation,Flatten
from tensorflow.keras import backend
import random

from tf_explain.utils.display import image_to_uint_255
import cv2
@tensorflow.function
def custom_get_gradients_and_filters(
    model, images, layer_name, class_index, use_guided_grads
):
    """
    Generate guided gradients and convolutional outputs with an inference.
    Args:
        model (tf.keras.Model): tf.keras model to inspect
        images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
        layer_name (str): Targeted layer for GradCAM
        class_index (int): Index of targeted class
        use_guided_grads (boolean): Whether to use guided grads or raw gradients
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: (Target layer outputs, Guided gradients)
    """
    grad_model = tensorflow.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tensorflow.GradientTape() as tape:
        inputs = tensorflow.cast(images, tensorflow.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    if use_guided_grads:
        grads = (
            tensorflow.cast(conv_outputs > 0, "float32")
            * tensorflow.cast(grads > 0, "float32")
            * grads
        )

    return conv_outputs, grads

def apply_heatmap_occlusion(
    heatmap, original_image, THRESHOLD=0.85
):
    """
    Apply a heatmap (as an np.ndarray) on top of an original image.
    Args:
        heatmap (numpy.ndarray): Array corresponding to the heatmap
        original_image (numpy.ndarray): Image on which we apply occlusion

    Returns:
        np.ndarray: Original image with heatmap applied
    """
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    heatmap = (heatmap - np.min(heatmap)) / (heatmap.max() - heatmap.min())

    super_threshold_indices = (heatmap>THRESHOLD)

    output = original_image
    output[super_threshold_indices] = 0.0
    return output

def batch_gradCAM_augmentationn(images,y_matrix,model, layer_name = 'mixed10'):
  contador = 0
  explainer = GradCAM()
  images_mod = images.numpy()

  for i in range(0,tensorflow.shape(images_mod)[0]):

    
    class_index = tensorflow.argmax(y_matrix[i])

    outputs, grads = custom_get_gradients_and_filters(
        model, np.expand_dims(images_mod[i], axis=0), layer_name, class_index, True
    )

    grid = explainer.explain((np.expand_dims(images_mod[i], axis=0),None), model, layer_name = layer_name, class_index= class_index)
    #explainer.save(grid, "./grad_cam_batch/", "heatmap_batch"+str(contador)+".jpg")
    cams = explainer.generate_ponderated_output(outputs, grads)
    heatmap = apply_heatmap_occlusion(cams[0].numpy(), images_mod[i])
    #explainer.save(image_to_uint_255(images_mod[i]), "./grad_cam_batch/", "original_batch"+str(contador)+".jpg")
    #explainer.save(image_to_uint_255(heatmap), "./grad_cam_batch/", "grad_cam_batch"+str(contador)+".jpg")
    #explainer.save(image_to_uint_255(images.numpy()[i]), "./grad_cam_batch/", "tensor"+str(contador)+".jpg")
    contador = contador+1
    #salvamos la imagen modificada
    images_mod[i]=heatmap
  images = tensorflow.convert_to_tensor(images_mod)
  return images,y_matrix

class CustomModel(tensorflow.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        n_random=random.random()
        P_Occlusion = 0.25
        #tensorflow.print("augmentation "+ str(n_random))

        if(n_random<=P_Occlusion):
          x,y = batch_gradCAM_augmentationn(x,y,self)

        with tensorflow.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

# Instantiate model
from tensorflow.keras.applications import InceptionV3
inceptionV3 = InceptionV3(weights='imagenet',include_top=False,input_shape=(img_height,img_width,3))
x = BatchNormalization()(inceptionV3.output)
x = Activation('relu')(x)
#x = AveragePooling2D(pool_size=3)(x)
x = Flatten()(x)
output = Dense(n_classes, activation='softmax', name="prediction_layer")(x)
model = CustomModel(inputs=[inceptionV3.input], outputs = output)
