import tensorflow as tf
from ops import batch_normal, de_conv, conv2d, fully_connect, lrelu
from utils import save_images, get_image, load_celebA, normalize
from utils import CelebA
import numpy as np
from commons import arch
from commons import measure
from commons.mnist.inf import inf_def
from commons.utils import get_inception_score
# import cv2
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.examples.tutorials.mnist import input_data
import os
import pdb
import time

TINY = 1e-8
d_scale_factor = 0.25
g_scale_factor = 1 - 0.75/2


class vaegan(object):

    #build model
    def __init__(self, batch_size, max_iters, repeat, load_type, latent_dim, log_dir, learnrate_init, mdevice, hparams, _lambda, data_ob, print_every,
                 save_every, ckp_dir):
        self.hparams = hparams
        self.mdevice = mdevice
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.print_every = print_every
        self.save_every = save_every
        self.repeat_num = repeat
        self.load_type = load_type
        self.data_ob = data_ob
        self.latent_dim = latent_dim
        self.log_dir = log_dir
        self.ckp_dir = ckp_dir
        self.learn_rate_init = learnrate_init
        self.log_vars = []
        self.alpha = _lambda[0]
        self.beta = _lambda[1]
        self.gamma = _lambda[2]
        self.channel = 3
        self.output_size = 64
        self.theta_ph = mdevice.get_theta_ph(hparams)
        self.theta_ph_rec = mdevice.get_theta_ph(hparams)
        self.theta_ph_xp = mdevice.get_theta_ph(hparams)
        self.images = tf.placeholder(tf.float32, shape=[
                                     None, self.output_size, self.output_size, self.channel], name="Inputs")
        # self.images = tf.reshape(self.input, [-1, 28, 28, 1])
        # self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.ep = tf.random_normal(shape=[self.batch_size, self.latent_dim])
        self.zp = tf.random_normal(shape=[self.batch_size, self.latent_dim])
        self.next_x, self.training_init_op = self.make_dataset(
            data_ob.train_data_list)
        self.next_x_val, self.val_init_op = self.make_dataset(
            data_ob.val_data_list)

    def make_dataset(self, data_list):
        dataset = tf.data.Dataset.from_tensor_slices(
            convert_to_tensor(data_list, dtype=tf.string))
        dataset = dataset.map(lambda filename: tuple(tf.py_func(self._read_by_function,
                                                                [filename], [tf.double])), num_parallel_calls=16)
        dataset = dataset.repeat(self.repeat_num)
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(self.batch_size))

        iterator = tf.data.Iterator.from_structure(
            dataset.output_types, dataset.output_shapes)
        next_x = tf.squeeze(iterator.get_next())
        init_op = iterator.make_initializer(dataset)
        return next_x, init_op

    def build_model_vaegan(self):

        self.x_lossy = arch.get_lossy(
            self.hparams, self.mdevice, self.images, self.theta_ph)

        self.z_mean, self.z_sigm = self.Encode(self.x_lossy)
        self.z_x = tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_sigm))*self.ep)
        self.x_tilde = self.generate(self.z_x, reuse=False)

        self.x_tilde_lossy = arch.get_lossy(
            self.hparams, self.mdevice, self.x_tilde, self.theta_ph_rec)

        self.l_x_tilde, self.De_pro_tilde = self.discriminate(
            self.x_tilde_lossy)
        # self.l_x_tilde, _ = self.discriminate(self.x_tilde, True)

        self.x_p = self.generate(self.zp, reuse=True)

        self.x_p_lossy = arch.get_lossy(
            self.hparams, self.mdevice, self.x_p, self.theta_ph_xp)

        self.l_x,  self.D_pro_logits = self.discriminate(self.x_lossy, True)
        _, self.G_pro_logits = self.discriminate(self.x_p_lossy, True)

        #KL loss
        self.kl_loss = self.KL_loss(
            self.z_mean, self.z_sigm)/(self.latent_dim*self.batch_size)

        # D loss
        self.D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.G_pro_logits), logits=self.G_pro_logits))
        self.D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_pro_logits) - d_scale_factor, logits=self.D_pro_logits))
        self.D_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.De_pro_tilde), logits=self.De_pro_tilde))

        # G loss
        self.G_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.G_pro_logits) - g_scale_factor, logits=self.G_pro_logits))
        self.G_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.De_pro_tilde) - g_scale_factor, logits=self.De_pro_tilde))
        # + self.D_tilde_loss
        self.D_loss = self.D_fake_loss + self.D_real_loss + self.D_tilde_loss

        # preceptual loss(feature loss)
        self.PL_loss = tf.reduce_mean(tf.reduce_sum(
            self.NLLNormal(self.l_x_tilde, self.l_x), [1, 2, 3])) / (4 * 4 * 256)
        L2_loss_1 = tf.reduce_mean(tf.reduce_sum(
            self.NLLNormal(self.x_tilde, self.x_lossy), [1, 2, 3])) / (64 * 64 * 3)
        L2_loss_2 = tf.reduce_mean(tf.reduce_sum(
            self.NLLNormal(self.x_tilde_lossy, self.x_lossy), [1, 2, 3])) / (64 * 64 * 3)
        self.L2_loss = L2_loss_1 if self.gamma == 0 else L2_loss_2
        #For encode
        # - self.LL_loss / (4 * 4 * 64)
        #self.kl_loss/(self.latent_dim*self.batch_size)-
        self.encode_loss = self.kl_loss - self.alpha * \
            self.L2_loss - self.beta * self.PL_loss

        #for Gen
        # - 1e-6*self.LL_loss
        #+ self.G_tilde_loss
        self.G_loss = self.G_fake_loss + self.G_tilde_loss
        self.recon_loss = tf.reduce_mean(tf.square(self.images - self.x_tilde))
        # self.recon_loss = tf.reduce_mean(tf.square(self.x_lossy - self.x_tilde))
        self.log_vars.append(("recon_loss", self.recon_loss))
        self.log_vars.append(("encode_loss", self.encode_loss))
        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))
        self.log_vars.append(("PL_loss", self.PL_loss))
        self.log_vars.append(("L2_loss", self.L2_loss))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.e_vars = [var for var in t_vars if 'e_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=3)
        self.summ = []
        for k, v in self.log_vars:
            self.summ.append(tf.summary.scalar(k, v))

    #do train
    def train(self):
        global_step = tf.Variable(0, trainable=False)
        add_global = global_step.assign_add(1)
        new_learning_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=global_step, decay_steps=10000,
                                                       decay_rate=0.98)
        #for D
        trainer_D = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_D = trainer_D.compute_gradients(
            self.D_loss, var_list=self.d_vars)
        opti_D = trainer_D.apply_gradients(gradients_D)

        #for G
        trainer_G = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_G = trainer_G.compute_gradients(
            self.G_loss, var_list=self.g_vars)
        opti_G = trainer_G.apply_gradients(gradients_G)

        #for E
        trainer_E = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_E = trainer_E.compute_gradients(
            self.encode_loss, var_list=self.e_vars)
        opti_E = trainer_E.apply_gradients(gradients_E)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            sess.run(init)
            # inception_score = 0
            # Initialzie the iterator
            sess.run(self.training_init_op)
            sess.run(self.val_init_op)
            summary_op = tf.summary.merge_all()
            # summary_op1 = tf.Summary(value=[tf.Summary.Value(tag="inc", simple_value=inception_score)])
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            summary_writer_train = tf.summary.FileWriter(
                '{}/{}/train'.format(self.log_dir, now), sess.graph)
            summary_writer_test = tf.summary.FileWriter(
                '{}/{}/test'.format(self.log_dir, now), sess.graph)
            step = 0
            if self.load_type != 'none' and os.path.exists(self.ckp_dir + '/' + self.load_type):
                ckpt = tf.train.get_checkpoint_state(
                    self.ckp_dir, latest_filename=self.load_type)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                    g_step = int(ckpt.model_checkpoint_path.split(
                        '/')[-1].split('-')[-1])
                    sess.run(global_step.assign(g_step))
                    print('model restored')

            step = global_step.eval()
            # test_images = load_celebA(self.data_ob,self.batch_size)
            test_images = sess.run(self.next_x_val)
            # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            # n_samples = mnist.train.num_examples
            # n_batches = n_samples//self.batch_size
            # test_images =_load_mnist(self.batch_size)
            # test_images = mnist.test.next_batch(self.batch_size)[0]
            # test_images = np.reshape(test_images,[-1,28,28,1])

            loss_list = []
            # inf_net = inf_def.InferenceNetwork()
            while step <= self.max_iters:
                next_x_images = sess.run(self.next_x)

                theta_val = self.mdevice.sample_theta(
                    self.hparams, self.batch_size)
                theta_val_rec = self.mdevice.sample_theta(
                    self.hparams, self.batch_size)
                theta_val_xp = self.mdevice.sample_theta(
                    self.hparams, self.batch_size)
                # next_x_images = sess.run(self.next_x)
                # next_x_images = np.reshape(next_x_images,[-1,28,28,1])
                fd = {self.images: next_x_images, self.theta_ph: theta_val,
                      self.theta_ph_rec: theta_val_rec, self.theta_ph_xp: theta_val_xp}
                sess.run(opti_E, feed_dict=fd)
                # optimizaiton G
                sess.run(opti_G, feed_dict=fd)
                # optimization D
                sess.run(opti_D, feed_dict=fd)
                # lossy_images , generated_image = sess.run([self.x_lossy,self.x_p], feed_dict=fd)

                if (step+1) % self.print_every == 0:

                    fd_test = {self.images: test_images,
                               self.theta_ph: theta_val, self.theta_ph_rec: theta_val_rec, self.theta_ph_xp: theta_val_xp}
                    tags = ['D_loss', 'G_loss', 'E_loss', 'PL_loss',
                            'L2_loss', 'kl_loss', 'recon_loss', 'Learning_rate']
                    all_loss_train = sess.run([self.D_loss, self.G_loss, self.encode_loss, self.PL_loss,
                                               self.L2_loss, self.kl_loss, self.recon_loss, new_learning_rate], feed_dict=fd)
                    all_loss_test = sess.run([self.D_loss, self.G_loss, self.encode_loss, self.PL_loss,
                                              self.L2_loss, self.kl_loss, self.recon_loss, new_learning_rate], feed_dict=fd_test)
                    print("Step %d: D: loss = %.7f G: loss=%.7f E: loss=%.7f PL loss=%.7f L2 loss=%.7f KL=%.7f RC=%.7f, LR=%.7f" % (
                        step, all_loss_train[0], all_loss_train[1], all_loss_train[2], all_loss_train[3],
                        all_loss_train[4], all_loss_train[5], all_loss_train[6], all_loss_train[7]))
                    summary_str = tf.Summary()
                    for k, v in zip(tags, all_loss_train):
                        summary_str.value.add(tag=k, simple_value=v)
                    summary_writer_train.add_summary(summary_str, step)
                    summary_str = tf.Summary()
                    for k, v in zip(tags, all_loss_test):
                        summary_str.value.add(tag=k, simple_value=v)
                    summary_writer_test.add_summary(summary_str, step)

                    # summary_str = sess.run(summary_op, feed_dict=fd_test)
                    # summary_writer_test.add_summary(summary_str, step)
                    # save_images(next_x_images[0:self.batch_size], [self.batch_size/8, 8],
                    #             '{}/train_{:02d}_real.png'.format(self.sample_path, step))

                    rec_images, lossy_images, generated_image, rc = sess.run(
                        [self.x_tilde, self.x_lossy, self.x_p, self.recon_loss], feed_dict=fd_test)
                    loss_list.append(rc)
                    # y_hat_val = inf_net.get_y_hat_val(rec_images)
                    # inception_score = get_inception_score(y_hat_val)
                    # score_list.append(inception_score)
                    # summary_str = sess.run(self.summ[0], feed_dict=fd_test)
                    # summary_str = sess.run(summary_op1)
                    if self.hparams.measurement_type == "blur_addnoise":
                        lossy_images = normalize(lossy_images)
                    if self.hparams.measurement_type == "drop_independent":
                        lossy_images = lossy_images * \
                            (1-self.hparams.drop_prob)
                    # summary_writer_test.add_summary(summary_str, step)
                    sample_images = [test_images[0:self.batch_size], lossy_images[0:self.batch_size], rec_images[0:self.batch_size],
                                     generated_image[0:self.batch_size]]
                    # save_images(sample_images[0:self.batch_size] , [self.batch_size/8, 8], '{}/train_{:02d}_recon.png'.format(self.sample_path, step))
                    titles = ['orig', 'lossy', 'reconstructed',
                              'generated', 'recon_loss']
                    save_images(sample_images, [self.batch_size/8, 8],
                                '{}/train_{:02d}_images.png'.format(self.log_dir, step), loss_list, titles)
                if (step+1) % self.save_every == 0:
                    self.saver.save(sess, self.ckp_dir + '/last.ckpt',
                                    global_step=global_step, latest_filename='last')
                    print("Model saved in file: %s" % self.ckp_dir)

                step += 1
                new_learn_rate = sess.run(new_learning_rate)
                if new_learn_rate > 0.00005:
                    sess.run(add_global)

    def test(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # Initialzie the iterator
            sess.run(self.training_init_op)

            sess.run(init)
            # self.saver.restore(sess, self.saved_model_path)

            next_x_images = sess.run(self.next_x)

            # real_images, sample_images = sess.run([self.images, self.x_tilde], feed_dict={self.images: next_x_images})
            sample_images = sess.run(self.x_p)
            save_images(sample_images[0:self.batch_size], [
                        self.batch_size/8, 8], '{}/train_{:02d}_{:04d}_con.png'.format(self.log_dir, 0, 0))
            # save_images(real_images[0:self.batch_size], [self.batch_size/8, 8], '{}/train_{:02d}_{:04d}_r.png'.format(self.sample_path, 0, 0))

            # ri = cv2.imread('{}/train_{:02d}_{:04d}_r.png'.format(self.sample_path, 0, 0), 1)
            # fi = cv2.imread('{}/train_{:02d}_{:04d}_con.png'.format(self.sample_path, 0, 0), 1)
            #
            # cv2.imshow('real_image', ri)
            # cv2.imshow('reconstruction', fi)
            #
            # cv2.waitKey(-1)

    def discriminate(self, x_var, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            conv1 = tf.nn.relu(conv2d(x_var, output_dim=32, name='dis_conv1'))
            conv2 = tf.nn.relu(batch_normal(
                conv2d(conv1, output_dim=128, name='dis_conv2'), scope='dis_bn1', reuse=reuse))
            conv3 = tf.nn.relu(batch_normal(
                conv2d(conv2, output_dim=256, name='dis_conv3'), scope='dis_bn2', reuse=reuse))
            conv4 = conv2d(conv3, output_dim=256, name='dis_conv4')
            middle_conv = conv4
            conv4 = tf.nn.relu(batch_normal(
                conv4, scope='dis_bn3', reuse=reuse))
            conv4 = tf.reshape(conv4, [self.batch_size, -1])

            fl = tf.nn.relu(batch_normal(fully_connect(
                conv4, output_size=256, scope='dis_fully1'), scope='dis_bn4', reuse=reuse))
            output = fully_connect(fl, output_size=1, scope='dis_fully2')

            return middle_conv, output

    def generate(self, z_var, reuse=False):

        with tf.variable_scope('generator') as scope:

            if reuse == True:
                scope.reuse_variables()

            d1 = tf.nn.relu(batch_normal(fully_connect(
                z_var, output_size=8*8*256, scope='gen_fully1'), scope='gen_bn1', reuse=reuse))
            d2 = tf.reshape(d1, [self.batch_size, 8, 8, 256])
            d2 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[
                            self.batch_size, 16, 16, 256], name='gen_deconv2'), scope='gen_bn2', reuse=reuse))
            d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[
                            self.batch_size, 32, 32, 128], name='gen_deconv3'), scope='gen_bn3', reuse=reuse))
            d4 = tf.nn.relu(batch_normal(de_conv(d3, output_shape=[
                            self.batch_size, 64, 64, 32], name='gen_deconv4'), scope='gen_bn4', reuse=reuse))
            d5 = de_conv(d4, output_shape=[
                         self.batch_size, 64, 64, 3], name='gen_deconv5', d_h=1, d_w=1)

            return tf.nn.tanh(d5)

    def Encode(self, x):

        with tf.variable_scope('encode') as scope:

            conv1 = tf.nn.relu(batch_normal(
                conv2d(x, output_dim=64, name='e_c1'), scope='e_bn1'))
            conv2 = tf.nn.relu(batch_normal(
                conv2d(conv1, output_dim=128, name='e_c2'), scope='e_bn2'))
            conv3 = tf.nn.relu(batch_normal(
                conv2d(conv2, output_dim=256, name='e_c3'), scope='e_bn3'))
            conv3 = tf.reshape(conv3, [self.batch_size, 256 * 8 * 8])
            fc1 = tf.nn.relu(batch_normal(fully_connect(
                conv3, output_size=1024, scope='e_f1'), scope='e_bn4'))
            z_mean = fully_connect(fc1, output_size=128, scope='e_f2')
            z_sigma = fully_connect(fc1, output_size=128, scope='e_f3')

            return z_mean, z_sigma

    def KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

    def NLLNormal(self, pred, target):

        c = -0.5 * tf.log(2 * np.pi)
        multiplier = 1.0 / (2.0 * 1)
        tmp = tf.square(pred - target)
        tmp *= -multiplier
        tmp += c

        return tmp

    def _parse_function(self, images_filenames):

        image_string = tf.read_file(images_filenames)
        image_decoded = tf.image.decode_and_crop_jpeg(
            image_string, crop_window=[218 / 2 - 54, 178 / 2 - 54, 108, 108], channels=3)
        image_resized = tf.image.resize_images(
            image_decoded, [self.output_size, self.output_size])
        image_resized = image_resized / 127.5 - 1

        return image_resized

    def _read_by_function(self, filename):

        array = get_image(filename, 108, is_crop=True, resize_w=self.output_size,
                          is_grayscale=False)
        real_images = np.array(array)
        return real_images
