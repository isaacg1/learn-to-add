import tensorflow as tf
import numpy as np
import random

sess = tf.InteractiveSession()

bits = 32

def to_bits(num):
    out = []
    for _ in range(bits):
        out.append(num & 1)
        num >>= 1
    return np.array(out[::-1])

def add_batch(elements):
    x = []
    y = []
    for _ in range(elements):
        a1 = random.randrange(1 << bits)
        a2 = random.randrange(1 << bits)
        b = (a1 + a2) & ((1 << bits) - 1)
        x.append(np.concatenate([to_bits(a1), to_bits(a2)]))
        y.append(to_bits(b))
    return np.array(x), np.array(y)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 2 * bits])
y_ = tf.placeholder(tf.float32, shape=[None, bits])

hidden_width = 6400

W_1 = weight_variable([2 * bits, hidden_width])
b_1 = bias_variable([hidden_width])

x_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

W_2 = weight_variable([hidden_width, bits])
b_2 = bias_variable([bits])
y = tf.matmul(x_1, W_2) + b_2

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.cast(y > 0, tf.float32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = add_batch(100)
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})
    if i % 1000 == 0:
        print(i, accuracy.eval(feed_dict={x:batch[0], y_:batch[1]}))


test_batch = add_batch(100)
print(accuracy.eval(feed_dict = {x: test_batch[0], y_: test_batch[1]}))
print_batch = add_batch(1)
result = y.eval(feed_dict = {x: print_batch[0]})
truth = print_batch[1]
print('\n'.join(map(str, zip(result[0], truth[0]))))
