import tensorflow as tf

x = tf.placeholder('float', [None, 3]) # define the shape of placeholder
y = x * 2 # operational subgraph still remains the same

with tf.Session() as session:
    x_data = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    result = session.run(y, feed_dict={x: x_data}) # feed the data
    print(result)
