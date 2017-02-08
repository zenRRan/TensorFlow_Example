#逻辑回归
import tensorflow as tf
#tensorflow自带mnist
#60000行的训练数据集（mnist.train）和10000行的测试数据集（mnist.test）
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

#parameters
train_epochs = 25   #迭代次数
learning_rate = 0.01 #梯度下降所用的步长
batch_sizes = 100  #每次随机选择的批量数
display_steps = 1  #每输出的间隔

#x, y作为输入用
x = tf.placeholder(tf.float32, [None, 784])   #数组展开成一个向量，长度是 28x28 = 784，怎么展开现在不需要知道。
y = tf.placeholder(tf.float32, [None, 10])    #labels是0-9，[1,0,0,0,0,0,0,0,0,0] 表示0

#[None, 10] = [None, 784] * W + b ,因为W，b需要不算更新，所以用变量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#用softmax实现模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)

#损失可以用交叉熵计算:（tf.reduce_sum求和，tf.reduce_mean求平均）
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

#梯度下降法
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化所有变量
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    #Training
    for epoch in range(train_epochs):
        avg_cost = 0.
        total_batchs = int(mnist.train.num_examples / batch_sizes)
        for i in range(total_batchs):
             batch_xs, batch_ys = mnist.train.next_batch(batch_sizes) #mnist.train.next_batch随机选取一批
             _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
             avg_cost += c/total_batchs
        #print each epoch
        if (epoch+1) % display_steps == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    #Testing
    #tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值，但返回值是bool类型向量
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    #比如[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75
    accurcy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    #最后，我们计算所学习到的模型在测试数据集上面的正确率
    #都可以  print("Accurcy:", accurcy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Accurcy:", sess.run(accurcy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))






