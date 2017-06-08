import tensorflow as tf

x = tf.placeholder('float', None) # can hold any data of any shape
y = x * 2 # operational subgraph is defined here which can be individually computed

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]}) # we are feeding data to placeholder
    print(result)
