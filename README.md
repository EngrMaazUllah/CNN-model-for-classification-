# CNN-model-for-classification-
Classifying the Object using image processing techniques 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import tensorflow as tf
tf.compat.v1.set_random_seed(2019)
# designing the model 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = (96,96,3)) ,
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation = "relu") ,
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = "relu") ,
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation="relu"),      #Adding the Hidden layer
    tf.keras.layers.Dropout(0.4,seed = 2019),
    tf.keras.layers.Dense(256,activation ="relu"),
    tf.keras.layers.Dropout(0.2,seed = 2019),
    tf.keras.layers.Dense(3,activation = "softmax")   #Adding the Output Layer
])

model.summary()

from tensorflow.keras.optimizers import Adam  # using adam optimizer 
adam=Adam(lr=0.0001) # initial run rate as 0.0001
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics = ['acc']) #as the categorical
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

bs=8         #Setting batch size
train_dir = "E:/train"   #Setting training directory 
validation_dir = "E:/test"   #Setting testing directory
# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. ) 
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
# Flow training images in batches of 20 using train_datagen generator
#Flow_from_directory function lets the classifier directly identify the labels from the name of the directories the image lies in
train_generator=train_datagen.flow_from_directory(train_dir,batch_size=bs,class_mode='categorical',target_size=(96,96))
# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=bs,
                                                         class_mode  = 'categorical',
                                                         target_size=(96,96))


# fitting the model
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=150 // bs,
                    epochs=200,
                    validation_steps=50 // bs,
                    callbacks = reduce_lr,
                    verbose=2)

results = model.evaluate(validation_generator, batch_size=96)
print("test loss, test acc:", results)
model.save('newfyp_model_2.h5') # saving the model
# plotting the accuracy graph
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper left')
plt.show()
