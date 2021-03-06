# kaggle-Flower-Classification-on-TPU
This Project is based on Kaggle's Competition : Flower classification on TPU

## About Challenge
  - Classify 104 different types of flowers
  ![image](https://user-images.githubusercontent.com/29517840/154855326-f858b156-f90e-430e-85fc-884573247626.png)

## Data Description
Images are provided in TFRecord format, a container format frequently used in Tensorflow to group data files for optimal training performace. 
Each file contains the id, label and image.
 - 12753 training images
 - 3712 validation images
 - 7382 unlabeled test images

 Data is provided for resolutions:
  - (512 X 512 X 3)
  - (331 X 331 X 3)
  - (224 X 224 X 3)
  - (192 X 192 X 3)

## Pipeline: 
Refer to Kaggle Notebook : [here](https://www.kaggle.com/workab/flowers-classification/notebook) for the approach and implementation.

## Results

  | Model | Score|
  | :---: | :---: |
  | MobilenetV2 | 0.93745 |
  | DenseNet201 | 0.96971 |
  | EfficientNetB7 | 0.95150 |
  | (0.62)*DenseNet201 + (1-0.62)*EfficientNetB7 | 0.97058 |
  
## Strategy
- **Input Data Pipeline:**
  - tf.data pipeline with data augmentation
  - Added additional data
  - Data Augmentation done are:
    1. Spatial level Transforms:
        - Random flip left right, Random flip up down, Random crop
    2. Pixel level Transforms :
        - Random Saturation, Random Contrast, Adjust Gamma, Random Brightness
    3. Miscellaneous : 
        - Transform Rotation, Transform Shift, Transform Shear, Transform Zoom 
- **Model**
  - BaseModel with input image size of 224 X 224 and Added GlobalAveragePooling layer
  - Transfer learning over several models out which DenseNet201 performed best   
  - Ensemble Learning with 2 or more models
    - used DenseNet201 and EfficientnetB7 :
        - instead of one network, we train two and then combine their probability distributions.

- **Training parameters**
  - Hyper-parameters : Image Size, Batch Size, Epochs, Learning rate, Types and Amount of Augmentation
  - used learning rate scheduling, for more stable training.
      - High initial learning rates can make loss explode. It is usually better linearly to increase learning rate from very small value over the first ~5 iterations.
        ```
        LR_START = 0.00001
        LR_MAX = 0.00005 * strategy.num_replicas_in_sync
        ```
        ![image](https://user-images.githubusercontent.com/29517840/154858360-c470ec56-df9b-433a-af53-cb6c13b74d71.png)
   - optimizer = Adam 
   - loss function = sparse_categorical_crossentropy
   - metrics = sparse_categorical_accuracy

## Predictions on real images

|marigold|rose|sunflower|dahlia|
| :-------------------------: | :-------------------------: | :-------------------------: | :-------------------------: |
| ![1](Predictions/marigold_IMG_20211227_114656.jpg) | ![rose](Predictions/rose_IMG_20220114_080043.jpg) |![sunflower](Predictions/sunflower_IMG_20200611_081818.jpg)|![dahlia](Predictions/pink_yellow_dahlia_IMG_20220108_145213.jpg)


## Extended my Work to build Android application for Flower classification
  - Converted trained model to TFLite Format
  - Did setup for Android App in Android studio
  - apk files are present [here](https://github.com/ankita-2015/kaggle-Flower-Classification-on-TPU/tree/main/AndroidApp/APKs)
