#　卷积神经网络
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend_config import epsilon

if __name__ == "__main__":
    (train_x,train_y),(test_x,test_y)=tf.keras.datasets.fashion_mnist.load_data()
    model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    model.compile(optimizer="adam",loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=["accuracy"])
    model.fit((train_x/255).reshape(-1,28,28,1),train_y,epochs=2,)

    model.evaluate((test_x/255).reshape(-1,28,28,1),test_y)

    
    print(test_y[0])
    print(np.argmax(model.predict((test_x[0]/255).reshape(-1,28,28,1))))
    plt.imshow(test_x[0])
