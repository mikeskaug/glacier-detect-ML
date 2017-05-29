import tensorflow as tf
import numpy as np


def get_batch(data, labels, size):
    indices = np.random.randint(0, data.shape[0], size)
    return data[indices, :], labels[indices, :]


def logistic_regression_model(data, labels):
    x = tf.placeholder(tf.float32, [None, data.shape[1]])
    W = tf.Variable(tf.zeros([data.shape[1], labels.shape[1]]))
    b = tf.Variable(tf.zeros([labels.shape[1]]))

    # The linear model y = W * x + b with softmax() to turn it into probabilities across the categories
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y


def loss(logits, labels):
    # The known values
    y_ = tf.placeholder(tf.float32, [None, labels.shape[1]])
    # loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(logits), reduction_indices=[1]))
    # print out cross entropy value to track convergence
    cross_entropy = tf.Print(cross_entropy, [cross_entropy], "CrossE", first_n=50)
    return cross_entropy


def train(loss, learning_rate):
    # how to train/optimize the model
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return train_step


def do_eval(sess):
    test_set = np.load('test_set.npy')
    test_set_shift = test_set - test_set.min(axis=0)
    test_set_norm = test_set_shift / test_set_shift.max(axis=0)
    test_labels = np.load('test_labels.npy')

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: test_set_norm, y_: test_labels}))


def run_training(unused_argv):
    training_set = np.load('training_set.npy')
    training_set_shift = training_set - training_set.min(axis=0)
    training_set_norm = training_set_shift / training_set_shift.max(axis=0)
    training_labels = np.load('training_labels.npy')

    logits = logistic_regression_model(training_set_norm, training_labels)
    loss_op = loss(logits, training_labels)
    train_op = train(loss_op, 0.8)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = get_batch(training_set_norm, training_labels, 500)
        sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 100 == 0:
            do_eval(sess)


if __name__ == "__main__":
    tf.app.run(main=run_training)
