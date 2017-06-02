import tensorflow as tf


x = tf.constant([35, 40, 45], name='x') # since this is a list now every element of list will participate in the operations performed on the list
y = tf.Variable(x + 5, name='y')


model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(y)) # [40 45 50]
