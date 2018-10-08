import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def test_network(inn, ans):
    #global
    #with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
    mine = sess.run(L2, feed_dict={xs:inn})
    error = tf.equal(tf.arg_max(mine,1), tf.arg_max(ans,1))
    acc = sess.run(tf.reduce_mean(tf.cast(error,tf.float32)))
    return acc

def Add_parameters(input, in_size, out_size, act_fuc=None):
    #in_size :  the size of input vector
    #out_size:  the size of output vector
    #act_fuc:   the activate function of the cell
    #input:     the input tensor
    Weight = tf.Variable(tf.random_normal([in_size,out_size]))
    bias = tf.Variable( tf.zeros([1,out_size])+0.1)
    result = tf.matmul(input,Weight) + bias
    if act_fuc is None:
        return result
    else:
        return act_fuc(result)

if __name__=='__main__':
   # print('exut')
    xs = tf.placeholder(tf.float32, [None,784])
    ys = tf.placeholder(tf.float32, [None,10])
    L1 = Add_parameters(xs, 784, 100, tf.nn.sigmoid)
    L2 = Add_parameters(L1, 100, 10, tf.nn.softmax)
    #loss = tf.reduce_mean(tf.square(ys - L2))
    loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(L2), reduction_indices=[1]) )
    solve = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    INIT = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(INIT)
    for i in range(10000):
        batch_x,batch_y = mnist.train.next_batch(50)
        sess.run(solve, feed_dict={xs:batch_x, ys:batch_y})
        if i%100 == 0:
            print("loss = ", sess.run(loss,feed_dict={xs:batch_x, ys:batch_y}))
            print(test_network(mnist.test.images, mnist.test.labels),'?')
    sess.close()


