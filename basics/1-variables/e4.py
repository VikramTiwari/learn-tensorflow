import tensorflow as tf

x = tf.constant(35, name='x')
print(x)
y = tf.Variable(x + 5, name='y')

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/tmp/tensorflow/", session.graph) # write session to a file which can be used by tensorboard to create a visualization of the model itself
    model =  tf.global_variables_initializer()
    session.run(model)
    print(session.run(y))

# run tensorboard (visualization platform for tensorflow):
# tensorboard --logdir=/tmp/tensorflow/
# open http://localhost:6006
