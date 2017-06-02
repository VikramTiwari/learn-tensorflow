import numpy as np
import tensorflow as tf


x = tf.constant(np.random.randint(1000, size=10000), name='x') # use numpy for large numerical operations
y = tf.Variable(x + 5, name='y')


model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(y))
