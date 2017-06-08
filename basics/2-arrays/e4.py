# mirror: where the first half of the image is copied, flipped (l-r) and then copied into the second half

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
    half_left = tf.slice(image, [0, 0, 0],
                         [shape[0], int(shape[1] / 2), shape[2]])
    half_right = tf.reverse_sequence(
        half_left, [int(shape[1] / 2)] * shape[0], 1, batch_dim=0)
    x = tf.concat([half_left, half_right], 1)
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()
