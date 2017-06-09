# age old

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path, 'MarshOrchid.jpg')
raw_image_data = mpimg.imread(filename)

image = tf.placeholder('uint8', [None, None, 3])
grayscale = tf.reduce_mean(image, axis=2) # take reduced mean

with tf.Session() as session:
    result = session.run(grayscale, feed_dict={
        image: raw_image_data
    })  # run the slice subgraph by feeding it image data
    print(result.shape)

plt.imshow(result)
plt.show()
