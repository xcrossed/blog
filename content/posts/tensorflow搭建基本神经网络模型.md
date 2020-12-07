#［tensorflow认证考试复习］01 构建和训练一个神经网络模型(Build and train neural network models using TensorFlow 2.x)

本期文章是一个系列课程，文章标题如下
(1)Build and train neural network models using TensorFlow 2.x
(2)Image classification
(3)Natural language processing(NLP)
(4)Time series, sequences and predictions

## Build and train neural network models using TensorFlow 2.x

使用tensorflow构建和训练一个神经网络模型

## 环境

tensorflow认证是基于tensorflow 2.x进行的，所以复习的时候一定要用这个版本的tensorflow进行复习

## 训练神经网络模型的步骤

### 准备数据集

本次使用mnist数据集

``` python
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
print(x_train.shape)
print(y_train.shape)
```

输出
(60000, 28, 28)
(60000, )

### 数据预处理

因为图像是 0~255 之间的整形，需要进行归一化到 0~1 区间，这个目的为了使模型精度更高．01区间收敛速度更快．具体这块可以单独展开讲．

``` python
x_train,x_test=x_train/255.0,x_test/255.0
```

### 搭建模型

模型这块使用的是顺序模型
第一层是平坦化，将二维变成一维，作为输入
第二层是普通的Dense层, 激活函数为relu
第三层是一个Drop层，防止过拟合的, drop的比例为0.2
第四层是一个输出层，激活函数为softmax，输出多个分类的概率值

``` python
model=tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(128,activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10,activation="softmax")
        ]
    )
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",
metrics=["accuracy"])
```

### 训练

``` python
history=model.fit(x_train,y_train,epochs=10)
```

###　评估

``` python
model.evaluate(x_test,y_test)
```

## 输出

``` shell
Epoch 1/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2974 - accuracy: 0.9141
Epoch 2/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1439 - accuracy: 0.9574
Epoch 3/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.1076 - accuracy: 0.9684
Epoch 4/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0858 - accuracy: 0.9736
Epoch 5/10
1875/1875 [==============================] - 3s 1ms/step - loss: 0.0738 - accuracy: 0.9764
Epoch 6/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.0644 - accuracy: 0.9793
Epoch 7/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.0559 - accuracy: 0.9820
Epoch 8/10
1875/1875 [==============================] - 3s 1ms/step - loss: 0.0532 - accuracy: 0.9826
Epoch 9/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0483 - accuracy: 0.9839
Epoch 10/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.0420 - accuracy: 0.9862
2020-12-06 22:38:52.602650: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 31360000 exceeds 10% of free system memory.
313/313 [==============================] - 0s 968us/step - loss: 0.0680 - accuracy: 0.9797
{'loss': [0.2974238991737366, 0.14386004209518433, 0.10755111277103424, 0.08577871322631836, 0.07378625124692917, 0.06444691121578217, 0.055944059044122696, 0.05322827026247978, 0.04829448089003563, 0.04199640452861786], 'accuracy': [0.9140833616256714, 0.9573666453361511, 0.9684333205223083, 0.973633348941803, 0.9763833284378052, 0.9793499708175659, 0.9820166826248169, 0.9825666546821594, 0.9839333295822144, 0.9861666560173035]}
```

##　总结
主要是熟悉了基于tensorflow 2.x基于自带数据集如何搭建一个简单的机器学习模型．
上面的是一个手写数字识别的数据集，应该是一个图像识别分类任务．
从最终结果来看，在测试集中的准确率低于训练集，有一定程度上的过拟合．

主要掌握就是顺序模型的搭建，至于损失函数，参数归一化，优化器，学习率，dropout, pooling，激活函数，反向传播，正向传播等知识，需要单独去学习．

### 完整代码

https://github.com/xcrossed/blog/blob/master/code/tf/01.py
