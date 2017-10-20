import tensorflow as tf
import time


session = tf.InteractiveSession()

w1= []
w2 = []
tf_matmul = []
for gid in range(4):
    with tf.device('/gpu:%d'%gid):
        w1.append(tf.random_uniform(minval=-1.0, maxval=1.0,shape=[10000,15000],dtype=tf.float64))
        w2.append(tf.random_uniform(minval=-1.0, maxval=1.0,shape=[15000,10000],dtype=tf.float64))


        tf_matmul.append(tf.matmul(w1[-1],w2[-1]))



for _ in range(10000):

    for gid in range(4):
        session.run(tf_matmul[gid])

    time.sleep(2)
