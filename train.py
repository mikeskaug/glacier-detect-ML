import tensorflow as tf
import numpy as np
import os


def placeholders(num_features, num_categories):
    data_placeholder = tf.placeholder(tf.float32, [None, num_features])
    labels_placeholder = tf.placeholder(tf.float32, [None, num_categories])
    return data_placeholder, labels_placeholder


def get_batch(data, labels, size):
    indices = np.random.randint(0, data.shape[0], size)
    return data[indices, :], labels[indices, :]


def logistic_regression_model(data_placeholder, num_features, num_categories):
    W = tf.Variable(tf.zeros([num_features, num_categories]))
    b = tf.Variable(tf.zeros([num_categories]))

    # The linear model y = W * x + b with softmax() to turn it into probabilities across the categories
    y = tf.nn.softmax(tf.matmul(data_placeholder, W) + b)
    return y


def loss(labels_placeholder, logits):
    # loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels_placeholder * tf.log(logits), reduction_indices=[1]))
    # print out cross entropy value to track convergence
    cross_entropy = tf.Print(cross_entropy, [cross_entropy], "CrossE", first_n=50)
    return cross_entropy


def train(loss, learning_rate):
    # how to train/optimize the model
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return train_step


def do_eval(sess, logits, data_placeholder, labels_placeholder, test_set, test_labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={data_placeholder: test_set, labels_placeholder: test_labels}))


def load_data(directory):
    training_set = np.load(os.path.join(os.path.abspath(directory), 'training_set.npy'))
    training_set_shift = training_set - training_set.min(axis=0)
    training_set_norm = training_set_shift / training_set_shift.max(axis=0)
    training_labels = np.load(os.path.join(os.path.abspath(directory), 'training_labels.npy'))

    test_set = np.load(os.path.join(os.path.abspath(directory), 'test_set.npy'))
    test_set_shift = test_set - test_set.min(axis=0)
    test_set_norm = test_set_shift / test_set_shift.max(axis=0)
    test_labels = np.load(os.path.join(os.path.abspath(directory), 'test_labels.npy'))
    return (training_set_norm, training_labels, test_set_norm, test_labels)


def save_checkpoint(sess, out_dir):
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(os.path.abspath(out_dir), 'model.ckpt'))


def run_training(unused_argv):
    (training_set, training_labels, test_set, test_labels) = load_data('./output')
    num_features = training_set.shape[1]
    num_categories = training_labels.shape[1]

    data_placeholder, labels_placeholder = placeholders(num_features, num_categories)

    # Setup the graph operations
    logits = logistic_regression_model(data_placeholder, num_features, num_categories)
    loss_op = loss(labels_placeholder, logits)
    train_op = train(loss_op, 0.8)

    # Initialize the Tensor Flow variables and session
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(5000):
        batch_xs, batch_ys = get_batch(training_set, training_labels, 500)
        sess.run(train_op, feed_dict={data_placeholder: batch_xs, labels_placeholder: batch_ys})

        if i % 100 == 0:
            do_eval(sess, logits, data_placeholder, labels_placeholder, test_set, test_labels)

        if i % 1000 == 0:
            save_checkpoint(sess, './output')


if __name__ == "__main__":
    tf.app.run(main=run_training)
