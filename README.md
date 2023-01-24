# Pneumonia-Detection-with-CNN-arquitecture-with-Tensorflow
 Deep CNN model for detect Pneumonia on X-ray images with an accuracy of 98.80% 

# Loading the dataset

Loading the images from the directory using the image_dataset_from_directory function from Keras.
We scale the image to a size of (180,180), this function returns a tensor batch dataset of the image with a batch size of 32
with this function we also convert the image into grayscale, the reason for that is that these images are X-ray images so
in these cases, the color channel doesn't have much relevance to other features like blank intensity changes and edge detection.

![https://user-images.githubusercontent.com/86735728/184779028-182a19c1-2983-488c-a40d-13dd2d90b8be.png](https://user-images.githubusercontent.com/86735728/184779028-182a19c1-2983-488c-a40d-13dd2d90b8be.png)

# Optimization of memory and batch size for the training process

In this section, we apply an optimization to the amount of data flowing from the directories to the model per epoch,
in this case, we use caches and prefetch function to avoid bottle throat when fitting our model.

# Building the model

In this section we build our model, I apply a convolution filter each with a feature maps of 32 and a kernel of 5x5,
each convolution has a respective Max pooling to make our model have more balance in its dimensionality

```
adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
def cnn_model():
  Model = Sequential()
  Model.add(tf.keras.layers.Rescaling(1./255))

  Model.add(Conv2D(32,(3,3), padding = 'same', input_shape = (180, 180, 1),activation = 'relu' ))
  Model.add(MaxPooling2D( pool_size = (2,2), padding = 'same', strides=(2, 2)))
  Model.add(Dropout(0.2))

  Model.add(Conv2D(64,(3,3), activation = 'relu',kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
  Model.add(MaxPooling2D( pool_size = (2,2), padding = 'same', strides=(2, 2)))
  Model.add(Dropout(0.2))

  Model.add(Conv2D(128,(3,3), activation = 'relu', kernel_regularizer =tf.keras.regularizers.l2( l=0.01) ))
  Model.add(MaxPooling2D( pool_size = (2,2), padding = 'same', strides=(2, 2)))
  Model.add(Dropout(0.2))

  Model.add(Flatten())
  Model.add(Dense(256, activation = 'relu'))
  Model.add(Dense(1))
  Model.compile(optimizer= adam ,loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
  return Model

```

the next part of the model is the compilation, in this case, we're applying the Adam optimizer because our
problem is binary classification, in this case, we have a learning rate of 0.001.
The next one is the loss, in this case, I use ***BinaryCrossentropy*** with a label smoothing of 0.0
the last one is the metrics, in this case, I use the Binary Accuracy function of Keras in which I use a threshold of the judgment of 0.7

# Fitting the model and starting the training

In this section before starting the training, I applied an early stop callback to reduce overfitting without compromising
on model accuracy with patience of 5.

# Evaluating performance

After training the model, I analyze his performance by plotting the accuracy and loss per epoch for both, the training and the validation set.
In this first iteration, we see a slightly good performance using a pooling technique of MaxPooling2D and with a learning rate of 0.003
but we can still make the performance more stable and the graph smoother, so we applied more strides and made the learning rate 0.001.
After another evaluation, we see improvement in the model, but the validation loss start to diverge after the 30 epochs, so we applied l2 regularization to see how this could improve the results.

![https://user-images.githubusercontent.com/86735728/184778843-0434192e-f6b2-40e4-9984-d482a30f536b.png](https://user-images.githubusercontent.com/86735728/184778843-0434192e-f6b2-40e4-9984-d482a30f536b.png)

Finally, after the regularization we could stable the model and avoid the divergence of the validation loss,
in this case, the final results were:

- Training accuracy: 99.54%
- Validation accuracy: 98.80%
