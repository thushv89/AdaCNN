import tensorflow as tf
import numpy as np

tf_ph = tf.placeholder(dtype=tf.float32,shape=[5])
map_op = tf.map_fn(lambda x: x*2,tf_ph)

session = tf.InteractiveSession()

print(session.run(map_op,feed_dict={tf_ph:np.asarray([1,2,3,4,5])}))