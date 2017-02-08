# 变量加法
import tensorflow as tf

state = tf.Variable(0, name='counter')
#print state.name     //counter:0

one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        sess.run(update)
        print sess.run(state)
########################################
# 1
# 2
# 3
# 4
# 5
########################################