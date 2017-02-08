#  线性回归

import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt # 可视化

rng = np.random

#parameters
train_epoches = 2000
learning_rate = 0.01
display_stap = 50

#training data
X_trains = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y_trains = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_examples = X_trains.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(rng.randn(), name="Weight")
b = tf.Variable(rng.randn(), name="bias")

activation = tf.add(tf.mul(X, W), b)

cost = tf.reduce_sum(tf.pow(activation - Y, 2)/(2*n_examples))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoche in range(train_epoches):
        for (x, y) in zip(X_trains, Y_trains):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if epoche % display_stap == 0:
            print("Epoche:",'%04d' % (epoche+1), "cost=",\
                  '{:.9f}'.format(sess.run(cost, feed_dict={X: X_trains, Y: Y_trains})),\
                  "W=", sess.run(W), "b=", sess.run(b))
    print("Optimization finished")
    print("cost=", "{:.9}".format(sess.run(cost, feed_dict={X: X_trains, Y: Y_trains})), "b=", sess.run(b))
#可视化
    # plt.plot(X_trains, Y_trains, 'ro', label='Original data')
    # plt.plot(X_trains, sess.run(W) * X_trains + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()
######################################################################################################
# Epoche: 0001 cost= 11.177735329 W= -0.470282 b= 0.897355
# Epoche: 0051 cost= 0.079104908 W= 0.224147 b= 0.984524
# Epoche: 0101 cost= 0.078863017 W= 0.225661 b= 0.973635
# Epoche: 0151 cost= 0.078648910 W= 0.227085 b= 0.963395
# Epoche: 0201 cost= 0.078459404 W= 0.228423 b= 0.953764
# Epoche: 0251 cost= 0.078291707 W= 0.229682 b= 0.944707
# Epoche: 0301 cost= 0.078143217 W= 0.230866 b= 0.936188
# Epoche: 0351 cost= 0.078011796 W= 0.23198 b= 0.928177
# Epoche: 0401 cost= 0.077895448 W= 0.233027 b= 0.920642
# Epoche: 0451 cost= 0.077792421 W= 0.234013 b= 0.913555
# Epoche: 0501 cost= 0.077701196 W= 0.234939 b= 0.906889
# Epoche: 0551 cost= 0.077620447 W= 0.235811 b= 0.900619
# Epoche: 0601 cost= 0.077548914 W= 0.236631 b= 0.894722
# Epoche: 0651 cost= 0.077485591 W= 0.237401 b= 0.889176
# Epoche: 0701 cost= 0.077429503 W= 0.238127 b= 0.88396
# Epoche: 0751 cost= 0.077379823 W= 0.238808 b= 0.879054
# Epoche: 0801 cost= 0.077335820 W= 0.23945 b= 0.87444
# Epoche: 0851 cost= 0.077296853 W= 0.240053 b= 0.870099
# Epoche: 0901 cost= 0.077262312 W= 0.240621 b= 0.866018
# Epoche: 0951 cost= 0.077231735 W= 0.241154 b= 0.862178
# Epoche: 1001 cost= 0.077204630 W= 0.241656 b= 0.858568
# Epoche: 1051 cost= 0.077180594 W= 0.242128 b= 0.855171
# Epoche: 1101 cost= 0.077159338 W= 0.242572 b= 0.851977
# Epoche: 1151 cost= 0.077140450 W= 0.24299 b= 0.848972
# Epoche: 1201 cost= 0.077123724 W= 0.243383 b= 0.846147
# Epoche: 1251 cost= 0.077108890 W= 0.243752 b= 0.843489
# Epoche: 1301 cost= 0.077095740 W= 0.2441 b= 0.840989
# Epoche: 1351 cost= 0.077084102 W= 0.244427 b= 0.838638
# Epoche: 1401 cost= 0.077073731 W= 0.244734 b= 0.836426
# Epoche: 1451 cost= 0.077064581 W= 0.245023 b= 0.834347
# Epoche: 1501 cost= 0.077056430 W= 0.245295 b= 0.832391
# Epoche: 1551 cost= 0.077049203 W= 0.245551 b= 0.83055
# Epoche: 1601 cost= 0.077042811 W= 0.245791 b= 0.828821
# Epoche: 1651 cost= 0.077037118 W= 0.246017 b= 0.827194
# Epoche: 1701 cost= 0.077032059 W= 0.24623 b= 0.825662
# Epoche: 1751 cost= 0.077027559 W= 0.246431 b= 0.824221
# Epoche: 1801 cost= 0.077023581 W= 0.246619 b= 0.822869
# Epoche: 1851 cost= 0.077020027 W= 0.246796 b= 0.821595
# Epoche: 1901 cost= 0.077016905 W= 0.246962 b= 0.820397
# Epoche: 1951 cost= 0.077014104 W= 0.247119 b= 0.81927
# Optimization finished
# cost= 0.0770116597 b= 0.81823
######################################################################################################
