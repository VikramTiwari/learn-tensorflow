# break it up, bring it back together

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path, 'MarshOrchid.jpg')
raw_image_data = mpimg.imread(filename)

image = tf.placeholder('uint8', [None, None, 3])
height, width, depth = raw_image_data.shape
# create corners
top_left = tf.slice(image, [0, 0, 0], [int(height / 2), int(width / 2), -1])
top_right = tf.slice(image, [0, int(width / 2), 0], [int(height / 2), -1, -1])
bottom_left = tf.slice(image, [int(height / 2), 0, 0],
                       [-1, int(width / 2), -1])
bottom_right = tf.slice(image, [int(height / 2), int(width/2), 0], [-1, -1, -1])

with tf.Session() as session:
    top_left = session.run(top_left, feed_dict={image: raw_image_data})
    top_right = session.run(top_right, feed_dict={image: raw_image_data})
    bottom_left = session.run(bottom_left, feed_dict={image: raw_image_data})
    bottom_right = session.run(bottom_right, feed_dict={image: raw_image_data})
    x = tf.concat([
        tf.concat([top_left, top_right], 1), tf.concat(
            [bottom_left, bottom_right], 1)
    ], 0)  # join them together. 1 for linear, 0 for vertical
    result = session.run(x)
    print(result.shape)

plt.imshow(result)
plt.show()
