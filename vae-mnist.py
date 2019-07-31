import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from scipy.misc import imsave, imshow
import os

class MNIST_VAE:

	def __init__(self, params):

		self.params = params

		self.input = tf.placeholder(tf.float32, shape=[None, 784], name="Inputs")
		input = tf.reshape(self.input, [-1, 28, 28, 1])

		with tf.variable_scope("encoder"):
			output_encode = layers.conv2d(input, num_outputs=16, kernel_size=3, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			output_encode = layers.conv2d(output_encode, num_outputs=32, kernel_size=3, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
			output_encode = layers.flatten(output_encode)

		self.z_mean = layers.fully_connected(output_encode, num_outputs=self.params.n_z, activation_fn=None)
		self.z_stddev = layers.fully_connected(output_encode, num_outputs=self.params.n_z, activation_fn=None)

		samples = tf.random_normal([self.params.minibatch_size,self.params.n_z], 0, 1, dtype=tf.float32)

		sampled_z = self.z_mean + (self.z_stddev * samples)

		with tf.variable_scope("decoder"):
			output_decode = layers.fully_connected(sampled_z, num_outputs=output_encode.get_shape()[1].value, activation_fn=None)
			output_decode = tf.reshape(output_decode, [-1, 7, 7, 32])
			output_decode = layers.conv2d_transpose(output_decode, num_outputs=16, kernel_size=3, stride=2)
			output_decode = layers.conv2d_transpose(output_decode, num_outputs=1, kernel_size=3, stride=2, activation_fn=tf.nn.sigmoid)

			self._prediction = output_decode

		with tf.variable_scope("decoder", reuse=True):
			output_sample = layers.fully_connected(samples, num_outputs=output_encode.get_shape()[1].value, activation_fn=None)
			output_sample = tf.reshape(output_sample, [-1, 7, 7, 32])
			output_sample = layers.conv2d_transpose(output_sample, num_outputs=16, kernel_size=3, stride=2)
			output_sample = layers.conv2d_transpose(output_sample, num_outputs=1, kernel_size=3, stride=2, activation_fn=tf.nn.sigmoid)

			self._sampled_predictions = output_sample

		flat_pred = tf.reshape(self._prediction, [-1, 784])

		self.gen_loss = -tf.reduce_sum(self.input * tf.log(1e-10 + flat_pred) + (1. - self.input) * tf.log(1e-10 + 1 - flat_pred), 1)
		self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.square(self.z_stddev) - tf.log(tf.square(self.z_stddev)) - 1,1)

		self._loss = tf.reduce_mean(self.gen_loss + self.latent_loss)

		self._optimize = tf.train.AdamOptimizer(learning_rate=self.params.lr).minimize(self.loss)

	@property
	def loss(self):
		return self._loss

	@property
	def optimize(self):
		return self._optimize

	@property
	def prediction(self):
		return self._prediction

	@property
	def sample(self):
		return self._sampled_predictions

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img

def train(params):

	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	n_samples = mnist.train.num_examples
	n_batches = n_samples//params.minibatch_size

	if os.path.isdir("results"):
		pass
	else:
		os.makedirs("results")

	if os.path.isdir("results_samples"):
		pass
	else:
		os.makedirs("results_samples")

	with tf.Session() as sess:

		vae = MNIST_VAE(params)
		sess.run(tf.global_variables_initializer())
		for i in range(params.epochs):
			for j in range(n_batches):
				batch = mnist.train.next_batch(params.minibatch_size)[0]
				sess.run(vae.optimize, feed_dict={vae.input : batch})
				if j==(n_batches-1):
					print "Epoch : " + str(i) + ", Cost : " + str(sess.run(vae.loss, feed_dict={vae.input : batch}))
					generated_images = sess.run(vae.prediction, feed_dict={vae.input : batch})
					generated_images = generated_images.reshape([params.minibatch_size, 28, 28])
					imsave("results/"+str(i)+".jpg", merge(generated_images[:params.minibatch_size],[8,8]))
					generate_images = sess.run(vae.sample)
					generate_images = generate_images.reshape([params.minibatch_size, 28, 28])
					imsave("results_samples/"+str(i)+".jpg", merge(generate_images[:params.minibatch_size],[8,8]))

if __name__=='__main__':
	flags = tf.app.flags
	flags.DEFINE_float("lr", 1e-3, "Learning rate for VAE")
	flags.DEFINE_integer("epochs", 100, "Epochs for training")
	flags.DEFINE_integer("minibatch_size", 64, "Mini-batch size for training")
	flags.DEFINE_integer("n_z", 20, "Latent space dimension")
	params = flags.FLAGS

	train(params)