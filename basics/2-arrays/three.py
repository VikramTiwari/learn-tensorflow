import tensorflow as tf # import tensorflow
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path, "MarshOrchid.jpg")
image = mpimg.imread(filename)

x = tf.Variable(image, name='x') # load image into a tensorflow variable

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1, 0, 2]) # transpose the axis 0 and 1 while keeping 2 constant
    session.run(model)
    result = session.run(x) # run the session ans store result

plt.imshow(result) # plot the transpose result
plt.show()
