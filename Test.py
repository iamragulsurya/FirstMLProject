//Import the Libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

//Import the Fashion MNIST dataset

fashion_ds=tf.keras.datasets.fashion_mnist

(train_images, train_lables),(test_images, test_lables) = fashion_ds.load_data()

train_images.shape

train_lables.shape

test_images.shape


test_lables.shape

class_names=['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

//Preprocess the data

plt.figure()
plt.imshow(train_images[10])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(8,8))

for i in range(1,26):
  plt.subplot(5,5,i)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i])
  plt.xlabel(class_names[train_lables[i]])

plt.show()

//Build the model

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

//Compile the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

//Train the model

model.fit(train_images, train_lables, epochs=10)

//Evaluate accuracy

test_loss, test_acc = model.evaluate(test_images, test_lables, verbose=2)

print(test_acc)

//Make predictions

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

prediction = probability_model.predict(test_images)

prediction[10]

print(np.argmax(prediction[7654]))
