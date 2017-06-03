import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path, 'MarshOrchid.jpg')
image = mpimg.imread(filename)
height, width, depth = image.shape

x = tf.Variable(image, name='x')

model  = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1, 0, 2])
    new_height = width # after transpose these have have changed
    new_width = height # after transpose these have have changed
    x = tf.reverse_sequence(x, [new_width] * new_height , 1 , batch_dim=0) # use new height and width for reversing sequence
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()
