# Pneumonia-Detection-with-CNN-arquitecture-with-Tensorflwo
 Deep CNN model for detect Pneumonia on X-ray images with an accuracy of 98.80% base on a
 
# Loading the dataset

Loading the images from directorys using the image_dataset_from_directory function from Keras.
we scale the image to size of (180,180), this function returns a tensor batch dataset of the image with a batch size of 32
with this function we also convert the image into gray scale, the reason for that is because this image are X-ray images so 
in this cases the color channel doesn't have much relevant than other features like blank intensity changes and edge detection.

![image](https://user-images.githubusercontent.com/86735728/184775446-364b05b5-ca31-4717-990f-dbc1bb39759a.png)

# Optimization of memory and batch size for the training process
In this section we apply an optimizacion to the amount of data flowing from the directories to the model per epochs, 
in this case we use caches and prefetch function to avoid bottle throat when fitting our model.

# Building the model
In this section we build our model, I apply convolution filter each witha feature maps of 32 and a kernel of 5x5, 
each convolution have a respective Max pooling to make our model have more balance in his dimensionality
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

the next part of the model is the compilation, in this case we applying the Adam optimizer beauces our 
problem is binary clasification, in this case we a learning rate of 0.001.
the next one is the loss, in this case I use BinaryCrossentropy with a label smooothing of 0.0
the last one is the metrics, in this case I use the BinaryAccuracy function of keras in which a use a threshold of judgmnet of 0.7

# Fitting the model and start the training
In this section before start the training, I applying a early stop callback to reduce overfitting without compromising
on model accuracy with a patience of 5.

# Evaluating performance
After training the model I analyse his performance by plotting the accuracy adn loss per epochs for both, the training and the validation set.
In this first iteration we see a slighly good performance using a pooling technique of MaxPooling2D and with a learning rate of 0.003
but we can still make the performance more stable and the graph smoother so we applied more strides and making the learning rate to 0.001.
After another evaluation we see improvement on the model, but the validation loss start to diverge after the 30 epochs so we apply l2 regularization to see how this could improve the results.
Finally after the regularization we could stable the model and avoid the divergence of the validation loss, 
in this case the final results was:

- Training accuracy: 99.54%
- Validation accuracy: 98.80%
