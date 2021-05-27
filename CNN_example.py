import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Users\劳水成\Desktop\手写数字识别\新建文件夹\data', one_hot=True)


n_input = 784
n_output = 10
stddev = 0.1
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=stddev)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=stddev)),
    'wd1': tf.Variable(tf.random_normal([7*7*128, 1024], stddev=stddev)),
    'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=stddev))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=stddev)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=stddev)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=stddev)),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=stddev))
}


def conv_basic(_input, _w, _b, _keepratio):
    # 输入
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
    # 第一个卷积层
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # 第二个卷积层
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)

    _densel = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])

    # 第一个全连接层
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_densel, _w['wd1']), _b['bd1']))
    _fc1_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # 第二个全连接层
    _out = tf.add(tf.matmul(_fc1_dr1, _w['wd2']), _b['bd2'])
    # 返回
    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
           'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'densel': _densel,
           'fc1': _fc1, 'fc_dr1': _fc1_dr1, 'out': _out
    }
    return out



x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)



_pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred, labels=y))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.global_variables_initializer()

# 保存
save_step = 1
saver = tf.train.Saver(max_to_keep=3)


do_train = 1
sess = tf.Session()
sess.run(init)

training_epochs = 10
batch_size = 100
display_step = 3

# 优化
if do_train == 1:
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        # 迭代
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio: 0.7})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.}) / total_batch

        if epoch % display_step == 0:
            print('Epoch: %03d/%03d cost: %.9f' % (epoch, training_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.})
            print('TRAIN ACCURACY: ', train_acc)


        if epoch % save_step == 0:
            saver.save(sess, r'save/nets/cnn_mnist_basic.ckpt-' + str(epoch))
if do_train == 0:
    epoch = training_epochs-1
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    saver.restore(sess, r'save/nets/cnn_mnist_basic.ckpt-' + str(epoch))
    test_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.})
    print('TEST ACCURACY: ', test_acc)
print('DONE!')



