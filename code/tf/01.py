import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.engine.sequential import relax_input_shape
from tensorflow.python.ops.gen_nn_ops import lrn

if __name__ == "__main__":
    print(tf.__version__)
    # load dataset
    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
    #data preparse process
    print(x_train.shape)
    print(y_train.shape)
    x_train,x_test=x_train/255.0,x_test/255.0
    # create model
    model=tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(128,activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10,activation="softmax")
        ]
    )
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]   )
    history=model.fit(x_train,y_train,epochs=10)
    model.evaluate(x_test,y_test)
    print(history.history)

    x=list(range(len(history.history["loss"])))
    y_loss=history.history["loss"]
    y_acc=history.history["accuracy"]

    plt.figure()
    plt.plot(x,y_loss,"r",label="loss")
    plt.plot(x,y_acc,"b",label="acc")
    plt.legend()
    plt.show()

