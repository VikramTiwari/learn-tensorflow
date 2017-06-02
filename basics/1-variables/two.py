import tensorflow as tf

x = tf.constant(35, name='x')
y = tf.Variable(x + 5, name = 'y')

# magic
model = tf.global_variables_initializer()

# create a session where actual events will happen
with tf.Session() as session:
    # run the model in session
    session.run(model)
    # run the session nad pass it y to compute
    print(session.run(y))
