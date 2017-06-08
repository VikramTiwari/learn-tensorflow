# flipud: flip image top to bottom

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path, 'MarshOrchid.jpg')
image = mpimg.imread(filename)

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    shape = session.run(tf.shape(image, name='shape'))
    x = tf.transpose(x, perm=[1, 0, 2]) # transpose
    x = tf.reverse_sequence(x, [shape[0]] * shape[1], 1, batch_dim=0) # reverse sequence
    x = tf.transpose(x, perm=[1, 0, 2]) # again transpose
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()
