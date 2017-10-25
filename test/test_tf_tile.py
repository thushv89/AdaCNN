import tensorflow as tf

session = tf.InteractiveSession()

a = tf.convert_to_tensor([[1],[2],[3]],dtype=tf.float32)

b = tf.tile(a,[1,3])

print(session.run(b))