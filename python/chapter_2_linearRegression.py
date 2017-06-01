# -*- coding:utf8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 1000
vectors_set = []

for i in xrange(num_points) :
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# plt.plot(x_data, y_data, 'ro')
# plt.show()



# tensorflow graph

W = tf.Variable(tf.random_uniform([1], -0.1, -0.1))
b = tf.Variable(tf.zeros(([1])))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# init = tf.global_variables_initializer()
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(8) :
    sess.run(train)
    print sess.run(W), sess.run(b)


# plt.plot(x_data, y_data, 'ro')
# plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

for step in xrange(5) :
    sess.run(train)
    print step, sess.run(W), sess.run(b)
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()