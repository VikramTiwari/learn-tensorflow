import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

# load the iamge
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path, 'MarshOrchid.jpg')
raw_image_data = mpimg.imread(filename)

image = tf.placeholder('uint8', [None, None, 3]) # to store image 3rd variable (depth) requires to be set rather than being any dimension. 3 because of RGB
slice = tf.slice(image, [1000, 0, 0], [3000, -1, -1]) # slice away 1000 pixels from top and 3000 from bottom

with tf.Session() as session:
    result = session.run(slice, feed_dict={image: raw_image_data}) # run the slice subgraph by feeding it image data
    print(result.shape)

plt.imshow(result)
plt.show()
