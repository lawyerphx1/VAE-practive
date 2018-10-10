import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

IMAGE_SIZE_MNIST = 28

def encoder(weights, biases, x):
    L1 = tf.nn.sigmoid( tf.matmul(x, weights['encoder_h1']) + biases['encoder_h1'] )
    L2 = tf.nn.sigmoid( tf.matmul(L1,weights['encoder_h2']) + biases['encoder_h2'] )
    return L2

def decoder(weights, biases, x):
    L1 = tf.nn.sigmoid( tf.matmul(x, weights['decoder_h1']) + biases['decoder_h1'] )
    L2 = tf.nn.sigmoid( tf.matmul(L1,weights['decoder_h2']) + biases['decoder_h2'] )
    return L2

def main(args):
    learning_rate = 0.01
#    training_epochs = 5
    batch_size = 256
    display_step = 1
#    examples_to_show = 10
    n_input = IMAGE_SIZE_MNIST ** 2
    n_hidden_1 = 256
    n_hidden_2 = 128
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_input]))
    }
    biases = {
        'encoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_input]))
    }
    X = tf.placeholder(tf.float32, [None,n_input])
    Y = decoder(weights,biases,encoder(weights,biases,X))
    loss = tf.reduce_mean(tf.square(X - Y))
    solve = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = int(mnist.train.num_examples / batch_size)
        for epoch in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([solve,loss], feed_dict={X:batch_xs})
            if epoch % display_step == 0:
                print("epoch:",epoch+1, "cost = ", c)


if __name__ == '__main__':
    main()