import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import scipy.ndimage
import numpy as np

x = tf.placeholder(tf.float32, [None, 784])# x belli bir değer değil.
                                           # Heseplama yapmak için gireceğimiz bir değer
										   
W = tf.Variable(tf.zeros([784, 10])) # Ağırlıklar 784x10 sıfır matris olarak tanımlıyoruz

b = tf.Variable(tf.zeros([10])) # b değişkenini tanımlıyoruz.
                                # b [1,10] sıfır matrisini tanımlıyoruz
								
y = tf.nn.softmax(tf.matmul(x, W) + b)# (x*W)+b işlemini yapıyor
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))#hata fonksiyonu

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)# kademeli düşürme yöntemine göre W ve b değerlerini bul

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()# değişkenlerimizi aktif ediyoruz

for _ in range(1000):# 1000 kere egitiyoruz
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#dogruluk oranı hesabı

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

cizim = np.vectorize(lambda x: 255 - x)(
        np.ndarray.flatten(scipy.ndimage.imread("resim.png", flatten=True)))
		
sonuc = sess.run(tf.argmax(y, 1), feed_dict={x: [cizim]})
print(sonuc)