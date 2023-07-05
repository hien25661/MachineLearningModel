# **Building Wrong Orientation Image Classification Model Tutorial**

In our previous blog, we explored the usage of YOLOv5 to detect frames within an image. Now, let's take the next step and build a model that can classify those frames as either having the correct orientation or being in the wrong orientation. The training process will be conducted using Kaggle.

## <a name="_jnztrr1558jr"></a>***What is Image Classification in Machine Learning?***
*Image classification is a computer vision task where label(s) are assigned to an entire image. The label should be representative of the main contents of the image. For instance, you could have a classifier that identifies whether a bird is, or is not, a certain species or sex…*

**Introduction of the problem:**

This poses a unique challenge that requires specialized techniques to handle wrong orientation images accurately. In this tutorial, we will guide you through the process of building a wrong orientation image classification by **using transfer learning from a pre-trained network,** enabling you to identify and classify images with incorrect orientations effectively. 


<a name="_hrc1k1flb6cr"></a> 

Correct Orientation 				Wrong Orientation

A pre-trained model is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. You either use the pretrained model as is or use transfer learning to customize this model to a given task.

In this notebook, you will try two ways to customize a pretrained model:

1. Feature Extraction: Use the representations learned by a previous network to extract meaningful features from new samples. You simply add a new classifier, which will be trained from scratch, on top of the pretrained model so that you can repurpose the feature maps learned previously for the dataset.

You do not need to (re)train the entire model. The base convolutional network already contains features that are generically useful for classifying pictures. However, the final, classification part of the pretrained model is specific to the original classification task, and subsequently specific to the set of classes on which the model was trained.

1. Fine-Tuning: Unfreeze a few of the top layers of a frozen model base and jointly train both the newly-added classifier layers and the last layers of the base model. This allows us to "fine-tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task.

You will follow the general machine learning workflow.

1. Examine and understand the data
1. Build an input pipeline.
1. Compose the model
   1. Load in the pretrained base model (and pretrained weights)
   1. Stack the classification layers on top
1. Train the model
1. Evaluate model	

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import os

import datetime

<a name="_itvtuswe53x7"></a>**Prerequisites**

- Basic knowledge of machine learning and computer vision concepts
- Python 3.x installed on your system.
- Pycharm IDE installed on your computer.
- Basic knowledge of Python programming and deep learning concepts.
- The Flickr8k dataset for correctly oriented images [(https://www.kaggle.com/datasets/adityajn105/flickr8k)](https://www.kaggle.com/datasets/adityajn105/flickr8k) 
- Kaggle Account for Model Trainin<a name="_ffcvjpou1sb8"></a>g

## <a name="_k3k3xltxip22"></a><a name="_ydgapy4lw8e"></a>**Dataset Preparation**
1. **Collecting Images**

Start by preparing the Flickr8k dataset, which will serve as the source of correctly oriented images. This dataset contains a collection of images along with their captions. We will only use the images for this tutorial. 

To generate wrong orientation images, we will apply rotation transformations to the original images.

You can see my Dataset here: <https://www.kaggle.com/datasets/hien240891/imagedirectionver2preprocess>
















**2. Image Data Pre-Processsing**

Login Kaggle, and create a new NoteBook






































###
- ### <a name="a-copy-dataset-and-splitting-dataset"></a>**Copy dataset and Splitting Dataset**
TRAIN\_VAL\_SPLIT=0.15

SEED=1337

IMG\_SIZE=(256, 256)

BATCH=16

train\_ratio = 0.85

test\_ratio = 0.15

train\_dir = '/kaggle/working/train'

test\_dir = '/kaggle/working/valid'

dir\_list = os.listdir("/kaggle/input/imagedirectionver2preprocess/archive\_shutil")

print('The Number of Classes in the raw is:{}'.format(len(dir\_list)))

source\_dir = "/kaggle/input/imagedirectionver2preprocess/archive\_shutil"

for folder in dir\_list:

`    `data\_dir = os.listdir(os.path.join(source\_dir, folder))

`    `np.random.shuffle(data\_dir)

`    `os.makedirs(os.path.join(train\_dir, folder), exist\_ok=True)

`    `os.makedirs(os.path.join(test\_dir, folder), exist\_ok=True)

`    `train\_data = data\_dir[: int(len(data\_dir) \* train\_ratio + 1)]

`    `test\_data = data\_dir[-int(len(data\_dir) \* test\_ratio):]

`    `for image in train\_data:

`        `copyfile(os.path.join(source\_dir, folder, image), os.path.join(train\_dir, folder, image))

`    `for image in test\_data:

`        `copyfile(os.path.join(source\_dir, folder, image), os.path.join(test\_dir, folder, image))

- ### <a name="btranform-data-with-label-for-each-class"></a>**Tranform data with label for each class**
\# Image data preprocessing

train\_dataset = tf.keras.utils.image\_dataset\_from\_directory(train\_dir,

`                                                            `shuffle=True,

`                                                            `batch\_size=BATCH,

`                                                            `image\_size=IMG\_SIZE)

validation\_dataset = tf.keras.utils.image\_dataset\_from\_directory(test\_dir,

`                                                                 `shuffle=True,

`                                                                 `batch\_size=BATCH,

`                                                                 `image\_size=IMG\_SIZE)

class\_names = train\_dataset.class\_names

print('Number of training batches: %d' % tf.data.experimental.cardinality(train\_dataset).numpy())

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation\_dataset).numpy())

Found 27510 files belonging to 2 classes.
Found 4854 files belonging to 2 classes.
Number of training batches: 1720
Number of validation batches: 304

We have 27510 images (1720 batches) for training and 4854 images (304 batches) for validation.
- ### <a name="c-dataset-visually"></a>**Dataset visually**
  - ### **Configure the dataset for performance**

Use buffered prefetching to load images from disk without having I/O become blocking. To learn more about this method see the [data performance](https://www.tensorflow.org/guide/data_performance) guide.

AUTOTUNE = tf.data.AUTOTUNE

train\_dataset = train\_dataset.prefetch(buffer\_size=AUTOTUNE)

validation\_dataset = validation\_dataset.prefetch(buffer\_size=AUTOTUNE)

- **Show the first 16 images and labels from the training set**

def plot\_images\_and\_labels(batch, predictions=None, stride=1, rows=4, cols=4):

`    `print(batch)

`    `plt.figure(figsize=(2 \* cols, 2 \* rows))

`    `for images, labels in batch:

`        `for i in range(rows \* cols):

`            `ax = plt.subplot(rows, cols, i + 1)

`            `# display every nth image in the batch

`            `idx = stride \* i

`            `image = images[idx]

`            `label = labels[idx].numpy()

`            `plt.imshow(image.numpy().astype("uint8"))

`            `title = "Label: " + str(label)

`            `title\_color = 'black'

`            `if predictions is not None:

`                `prediction = predictions[idx]

`                `title += f", Pred: {prediction:.2f}"

`                `error = np.abs(label - prediction)

`                `if error > 0.5:

`                    `title\_color = 'red'

`            `plt.title(title, color=title\_color)

`            `plt.axis("off")

`        `break

`    `plt.show()

plot\_images\_and\_labels(train\_dataset.take(1), stride=1)

- Positive sample: Wrong Orientation Images , label=1.
- Negative sample: Correct Orientation Images,  label=0.



##
## <a name="_hu1sevgle8xg"></a><a name="_snepr2ybbjum"></a>**Training**
1. ## **Create the base model from the pre-trained Model**
You will create the base model from the EfficientNetB0 model. This model is also pre-trained on the ImageNet dataset, which comprises 1.4 million images and 1000 classes.

ImageNet provides a diverse range of categories such as jackfruit and syringe. This extensive knowledge base will assist in classifying Wrong  and Correct Orientation from our specific dataset. 

First, you need to determine the layer of EfficientNetB0 to use for feature extraction.The last classification layer at the top is not as useful. Instead, it is common practice to utilize the last layer before the flatten operation, known as the "bottleneck layer." The features from the bottleneck layer exhibit greater generality compared to the final/top layer. 

To implement this, begin by instantiating an EfficientNetB0 model with pre-loaded weights trained on ImageNet. By specifying the include\_top=False argument, you load a network that doesn't incorporate the classification layers at the top. This configuration is well-suited for feature extraction purposes.


base\_model = tf.keras.applications.efficientnet.EfficientNetB0(weights="imagenet", include\_top=False,                                                       input\_tensor=tf.keras.layers.Input(shape=(256, 256, 3)))

1. ## **Feature extraction**
In this step, you will freeze the convolutional base created from the previous step and to use as a feature extractor. Additionally, you add a classifier on top of it and train the top-level classifier.
- ### ***Freeze the convolutional base***
It is important to freeze the convolutional base before you compile and train the model. Freezing (by setting layer.trainable = False) prevents the weights in a given layer from being updated during training. EfficientNetB0 has many layers, so setting the entire model's trainable flag to False will freeze all of them.

base\_model.trainable = False

- ### ***Important note about BatchNormalization layers***
Many models contain [tf.keras.layers.BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) layers. This layer is a special case and precautions should be taken in the context of fine-tuning, as shown later in this tutorial.

When you set ***layer.trainable = False***, the ***BatchNormalization*** layer will run in inference mode, and will not update its mean and variance statistics.

When you unfreeze a model that contains **BatchNormalization** layers in order to do fine-tuning, you should keep the **BatchNormalization** layers in inference mode by passing training = False when calling the base model. Otherwise, the updates applied to the non-trainable weights will destroy what the model has learned.

For more details, see the [Transfer learning guide](https://www.tensorflow.org/guide/keras/transfer_learning).

data\_augmentation = tf.keras.Sequential([

`  `tf.keras.layers.RandomFlip("horizontal"),

])

model = tf.keras.Sequential([

`    `base\_model,

`    `data\_augmentation,

`    `tf.keras.layers.GlobalMaxPooling2D(),

`    `tf.keras.layers.Flatten(),

`    `tf.keras.layers.Dense(128, activation='relu'),

`    `tf.keras.layers.BatchNormalization(),

`    `tf.keras.layers.Dropout(0.3),

`    `tf.keras.layers.Dense(64, activation='relu'),

`    `tf.keras.layers.BatchNormalization(),

`    `tf.keras.layers.Dropout(0.3),

`    `tf.keras.layers.Dense(32, activation='relu'),

`    `tf.keras.layers.Dropout(0.3),

`    `tf.keras.layers.Dense(1, activation='sigmoid')

])
- ### ***Compile the model***
Compile the model before training it. Since there are two classes, use the [***tf.keras.losses.BinaryCrossentropy***](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy) loss with ***from\_logits=True*** since the model provides a linear output.

model.compile(

`    `optimizer=tf.keras.optimizers.Adam(learning\_rate=2e-4),

`    `loss=tf.losses.BinaryCrossentropy(),

`    `metrics=[tf.keras.metrics.BinaryAccuracy(), 'AUC'])

model.summary()



Model: "sequential\_3"

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Layer (type)                 Output Shape              Param #   

\=================================================================

efficientnetb0 (Functional)  (None, 7, 7, 1280)        4049571   

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

sequential\_2 (Sequential)    (None, 7, 7, 1280)        0         

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

global\_max\_pooling2d\_1 (Glob (None, 1280)              0         

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

flatten\_1 (Flatten)          (None, 1280)              0         

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

dense\_4 (Dense)              (None, 128)               163968    

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

batch\_normalization\_2 (Batch (None, 128)               512       

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

dropout\_3 (Dropout)          (None, 128)               0         

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

dense\_5 (Dense)              (None, 64)                8256      

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

batch\_normalization\_3 (Batch (None, 64)                256       

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

dropout\_4 (Dropout)          (None, 64)                0         

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

dense\_6 (Dense)              (None, 32)                2080      

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

dropout\_5 (Dropout)          (None, 32)                0         

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

dense\_7 (Dense)              (None, 1)                 33        

\=================================================================

Total params: 4,224,676

Trainable params: 174,721

Non-trainable params: 4,049,955\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

1. ## **Train the Model**
!mkdir -p /kaggle/tmp

FILECACHE = False

DS\_CACHE\_FILE = "/kaggle/tmp/dataset.cache" if FILECACHE else ""

VAL\_CACHE\_FILE = "/kaggle/tmp/dataset\_val.cache" if FILECACHE else ""

\# Cache dataset to avoid running the CPU-intensive augmentation (image flipping) on every epoch. First epoch 

\# will be rather slow, but in the next iteration the cached data will be used.

train\_ds = dataset.cache(DS\_CACHE\_FILE).shuffle(32).prefetch(buffer\_size=tf.data.AUTOTUNE)

val\_ds = dataset\_val.cache(VAL\_CACHE\_FILE).prefetch(buffer\_size=tf.data.AUTOTUNE)

callbacks = [

`    `tf.keras.callbacks.ReduceLROnPlateau(monitor="val\_auc", factor=0.5, patience=8, 

`                                                  `cooldown=3, mode="max", verbose=1), 

`    `tf.keras.callbacks.EarlyStopping(monitor="val\_auc", patience=8, mode="max", verbose=1, restore\_best\_weights=True)

]

history = model.fit(

`    `train\_ds,

`    `validation\_data=val\_ds,

`    `epochs=30,

`    `callbacks = callbacks)

After training for 30 epochs, you should see ~97% accuracy on the validation set.

Epoch 20/30

100/100 [==============================] - 7s 71ms/step - loss: 0.0845 - binary\_accuracy: 0.9681 - auc: 0.9942 - val\_loss: 0.1935 - val\_binary\_accuracy: 0.9292 - val\_auc: 0.9770

Epoch 21/30

100/100 [==============================] - 7s 71ms/step - loss: 0.0860 - binary\_accuracy: 0.9677 - auc: 0.9940 - val\_loss: 0.2417 - val\_binary\_accuracy: 0.9219 - val\_auc: 0.9737

Epoch 22/30

100/100 [==============================] - 7s 72ms/step - loss: 0.0854 - binary\_accuracy: 0.9700 - auc: 0.9939 - val\_loss: 0.1989 - val\_binary\_accuracy: 0.9344 - val\_auc: 0.9766

Epoch 00022: ReduceLROnPlateau reducing learning rate to 9.999999747378752e-05.

Restoring model weights from the end of the best epoch.

Epoch 00022: early stopping
- ### ***Save the trained model:***

model.save('/kaggle/working/img\_orientation\_detector.h5')


1. ## **Evaluation**

Let's take a look at the learning curves of the training and validation accuracy/loss when using the EfficientNetB0  as a pre-trained model

def plot\_history(history):

`    `PLOTS = ["loss", "accuracy", "auc", "lr"]

`    `fig, axis = plt.subplots(1, len(PLOTS), figsize=(20, 6))

`    `for plot, ax in zip(PLOTS, axis.flatten()):

`        `for label, data in history.items():

`            `if plot in label:

`                `ax.plot(data, label=label)

`        `ax.legend()

`        `ax.grid()

`    `plt.tight\_layout()

`    `plt.show()



plot\_history(history.history)




## <a name="_uqj2s1mxf53l"></a>**Testing**
1. ## **Predict on Validation set**

Let's inspect the model's predictions visually to see how it performs. Pick a random selection of images from the validation set and plot them along with ground truth labels and predictions.



sampled\_images, sampled\_labels = validation\_dataset.unbatch().shuffle(1024).batch(64).take(1).get\_single\_element()

predictions = model.predict(sampled\_images)

plot\_images\_and\_labels([(sampled\_images, sampled\_labels)], predictions.squeeze(), stride=2, rows=5)




1. **Testing the trained model**

import cv2

PATH\_TO\_MODEL = '/kaggle/working/img\_orientation\_detector.h5'

img\_direction\_model = tf.keras.models.load\_model(PATH\_TO\_MODEL)

input\_img = cv2.imread("/kaggle/input/testing/image37.jpg", cv2.IMREAD\_COLOR)

\# input image size

width = 256

height = 256

dim = (width, height)

plt.figure(figsize=(4, 4))

\# resize image

resized = cv2.resize(input\_img, dim)

image\_tensor = tf.convert\_to\_tensor(resized)

image\_tensor = image\_tensor[tf.newaxis, ...]

print(image\_tensor.shape)

prediction = img\_direction\_model.predict(image\_tensor)

print(prediction)

plt.title(str(prediction[0][0]))

plt.imshow(resized)

plt.show()


## <a name="_yfw46gl3650y"></a>**Conclusion**
In this tutorial, we walked through the process of building a wrong orientation image classification model using **using transfer learning from a pre-trained model**. By following these steps, you can effectively classify images with incorrect orientations, which can be useful in various applications that require image alignment and preprocessing. Remember to experiment with different model architectures, hyperparameters, and data augmentation techniques to further enhance the performance of your classification model.


