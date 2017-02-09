import tensorflow as tf

x_data = [1, 2, 3]
y_data = [2, 4, 6]

learning_rate = 0.1
train_steps = 1000

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

prediction = W*X + b

cost = tf.reduce_mean(tf.square(prediction - y_data))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps+1):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if i % 20 == 0:
            print("Step:", i, "Cost:", sess.run(cost, feed_dict={X: x_data, Y: y_data}), "W=", sess.run(W), "b=", sess.run(b))

    print(sess.run(prediction, feed_dict={X:4}))
    print(sess.run(prediction, feed_dict={X:3.6}))