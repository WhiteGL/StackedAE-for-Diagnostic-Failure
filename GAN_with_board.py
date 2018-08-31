import tensorflow as tf
import numpy as np
import data
import os
import shutil
import random

height = 1
width = 400
size = height * width


to_train = True
to_restore = False
output_path = "dfoutput"
log_dir = 'newboard'

max_epoch = 200
learning_rate = 0.0001


h1_size = 150
h2_size = 300
z_size = 100
batch_size = 20


real_data = data.normal_smp
real_data_batch = np.empty((batch_size, 400), dtype=float)
for i in range(batch_size):
    real_data_batch[i, ] = real_data[random.randint(1, 609), ]

sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x_data = tf.placeholder(tf.float32, [batch_size, size], name="x_data")
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)
    z_sample_val = np.random.normal(0, 1, size= (batch_size, z_size)).astype(np.float32)

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

      # 计算参数的标准差
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # 设置命名空间
    with tf.name_scope(layer_name):
        # 调用之前的方法初始化权重w，并且调用参数信息的记录方法，记录w的信息
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        # 调用之前的方法初始化权重b，并且调用参数信息的记录方法，记录b的信息
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
      # 执行wx+b的线性计算，并且用直方图记录下来
        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('linear', preactivate)
      # 将线性输出经过激励函数，并将输出也用直方图记录下来
            if act is None:
                activations = preactivate
            else:
                activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)

    return activations



# build generator
with tf.variable_scope('gen'):
    gl1 = nn_layer(z_prior, z_size, h1_size, 'g_l1', tf.nn.relu)
    gl2 = nn_layer(gl1, h1_size, h2_size, 'g_l2', tf.nn.relu)
    x_generated = nn_layer(gl2, h2_size, size, 'g_l3', tf.nn.tanh)

#build discriminor
with tf.variable_scope('dic'):
    x_in = tf.concat([x_data, x_generated], 0)
    dl1 = tf.nn.dropout(nn_layer(x_in, size, h2_size, 'd_l1', tf.nn.relu), keep_prob)
    dl2 = tf.nn.dropout(nn_layer(dl1, h2_size, h1_size, 'd_l2', tf.nn.relu), keep_prob)
    dl3 = nn_layer(dl2, h1_size, 1, 'd_l3', None)
    y_data = tf.nn.sigmoid(tf.slice(dl3, [0, 0], [batch_size, -1], name="y_data"))
    y_generated = tf.nn.sigmoid(tf.slice(dl3, [batch_size, 0], [-1, -1], name="y_generated"))

with tf.name_scope('g_loss'):
    g_loss = tf.reduce_mean(tf.log(1 - y_generated))

tf.summary.scalar('g_loss', g_loss)

with tf.name_scope('d_loss'):
    d_loss = - tf.reduce_mean(tf.log(y_data) + tf.log(1 - y_generated))

tf.summary.scalar('d_loss', d_loss)

with tf.name_scope('train'):
    train_step_d = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dic'))
    train_step_g = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen'))

with tf.name_scope('accuracy'):
    correct_prediction = y_data / y_generated
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

tf.global_variables_initializer().run()

steps = 600/batch_size

for i in range(sess.run(global_step), max_epoch):
    for j in np.arange(steps):
        print("epoch:%s, iter:%s" % (i, j))
        z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        # sess.run([train_step_d],
                 # feed_dict={x_data: real_data_batch, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
        # train_writer.add_summary(summary_d, j)
        if j % 1 == 0:
            summary_g, _, _ = sess.run([merged, train_step_g, train_step_d],
                     feed_dict={x_data: real_data_batch, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            train_writer.add_summary(summary_g, j)
    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
    # np.savetxt("dfoutput/sample{0}.csv".format(i), x_gen_val)
    z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
    # np.savetxt("dfoutput/random_sample{0}.csv".format(i), x_gen_val)
    sess.run(tf.assign(global_step, i + 1))
train_writer.close()