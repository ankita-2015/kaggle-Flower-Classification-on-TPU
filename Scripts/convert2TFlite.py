import tensorflow as tf
from tensorflow.keras.models import load_model
import pathlib

from tensorflow.keras import backend, layers
class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)
# model = load_model(r"EfficientNetB7.h5",custom_objects={'FixedDropout':FixedDropout(rate=0.2)})


model = load_model("EfficientNetB7.h5")

tflite_models_dir = pathlib.Path("TFlite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# unquantized model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# # Save the unquantized/float model:
tflite_model_file =  tflite_models_dir/"flower_classification_EfficientNetB7.tflite"
tflite_model_file.write_bytes(tflite_model)


# quantized model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()  

# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"flower_classification_EfficientNetB7_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)
