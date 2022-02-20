import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

IMAGE_SIZE = [224,224] 
CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102

print(f"number of flower classes: {len(CLASSES)}")

from tensorflow.keras import backend, layers
class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

def get_model():
    base_model = tf.keras.applications.MobileNetV2(weights=None, 
                                                include_top=False,
                                                input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = True
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    return model

# model = get_model()
# model.load_weights("MobileNetV2.h5")
  
files = glob(r"..\test_images\**\*.jpg",recursive=True)
model_name = "MobileNetV2.h5"

model = load_model(model_name)
# model = load_model(r"EfficientNetB7.h5",custom_objects={'FixedDropout':FixedDropout(rate=0.8)})

if not os.path.exists("Results/"+model_name):
    os.mkdir("Results/"+model_name)

def decode_image(image_data):
    image = cv2.imread(image_data)[:,:,::-1]/255.0
    image = cv2.resize(image,(224,224))
    return image

for i in files:
    image= decode_image(i)
    # image = image.reshape(1,512,512,3)
    image = np.expand_dims(image, axis=0) 
    pred  = model.predict(image)
    v = np.argmax(pred,axis=-1)
    
    print(v, CLASSES[v[0]])
    plt.imshow(image[0])
    plt.title(CLASSES[v[0]])
    plt.show()
    n = i.split("\\")[-1]
    cv2.imwrite(f"Results/{model_name}/{CLASSES[v[0]]}_{n}",image[0][:,:,::-1]*255.0)
