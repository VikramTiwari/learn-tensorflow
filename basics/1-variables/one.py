# import tensorflow module and use it as tensorflow
import tensorflow as tf

x = tf.constant(35, name='x') # create a variable in python which is a tensorflow constant with a name
y = tf.Variable(x + 5, name='y') # craete a variable in python which is a tensorflow variable with a name

print(y) # this is printing out y which is an equation representation rather than the result of the equation
