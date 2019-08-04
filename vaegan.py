import tensorflow as tf
from ops import batch_normal, de_conv, conv2d, fully_connect, lrelu
from utils import save_images, get_image, load_celebA, normalize, merge, compute_pnsr_ssim
from utils import CelebA
import numpy as np
from commons import arch
from commons import measure
from commons.mnist.inf import inf_def
from commons.utils import get_inception_score
# import cv2
from skimage import io
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.examples.tutorials.mnist import input_data
import os
import pdb
import time
import matplotlib.pyplot as plt
from model import Model
from dataloader import Dataloader
TINY = 1e-8
d_scale_factor = 0.25
g_scale_factor = 1 - 0.75/2


class vaegan(object):

    #build model
    def __init__(self, batch_size, max_iters, repeat, load_type, latent_dim, log_dir, learnrate_init, mdevice, _lambda, data_ob, print_every,
                 save_every, ckp_dir, flags):
        self.FLAGS = flags
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
        self.theta_ph = mdevice.get_theta_ph(flags)
        self.theta_ph_rec = mdevice.get_theta_ph(flags)
        self.theta_ph_xp = mdevice.get_theta_ph(flags)
        self.images = tf.placeholder(tf.float32, shape=[
                                     None, self.output_size, self.output_size, self.channel], name="Inputs")
        self.best_loss = np.inf
        # self.images = tf.reshape(self.input, [-1, 28, 28, 1])
        # self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.ep = tf.random_normal(shape=[self.batch_size, self.latent_dim])
        self.zp = tf.random_normal(shape=[self.batch_size, self.latent_dim])
        self.dataset = Dataloader(self.repeat_num, self.batch_size, self.output_size)
        self.next_x, self.training_init_op = self.dataset.make_dataset(
            data_ob.train_data_list)
        self.next_x_val, self.val_init_op = self.dataset.make_dataset(
            data_ob.val_data_list)
        self.M = Model(self.batch_size)
        
        np.random.seed(1)

    def build_model_vaegan(self):

        self.x_lossy = arch.get_lossy(
            self.FLAGS, self.mdevice, self.images, self.theta_ph)

        self.z_mean, self.z_sigm = self.M.Encode(self.x_lossy)
        self.z_x = tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_sigm))*self.ep)
        self.x_tilde = self.M.generate(self.z_x, reuse=False)
        if self.FLAGS.supervised:
            self.x_tilde_lossy = self.x_tilde
        else:
            self.x_tilde_lossy = arch.get_lossy(
                self.FLAGS, self.mdevice, self.x_tilde, self.theta_ph_rec)

        self.l_x_tilde, self.De_pro_tilde = self.M.discriminate(
            self.x_tilde_lossy)
        # self.l_x_tilde, _ = self.discriminate(self.x_tilde, True)

        self.x_p = self.M.generate(self.zp, reuse=True)
        if self.FLAGS.supervised:
            self.x_p_lossy = self.x_p
        else:
            self.x_p_lossy = arch.get_lossy(
                self.FLAGS, self.mdevice, self.x_p, self.theta_ph_xp)
            
        self.l_x,  self.D_pro_logits = self.M.discriminate(self.x_lossy
                                                           if not self.FLAGS.supervised else self.images, True)
        _, self.G_pro_logits = self.M.discriminate(self.x_p_lossy, True)

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
            self.NLLNormal(self.x_tilde, self.x_lossy 
            if not self.FLAGS.supervised else self.images), [1, 2, 3])) / (64 * 64 * 3)
        L2_loss_2 = tf.reduce_mean(tf.reduce_sum(
            self.NLLNormal(self.x_tilde_lossy, self.x_lossy 
            if not self.FLAGS.supervised else self.images), [1, 2, 3])) / (64 * 64 * 3)
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
        self.saver_best = tf.train.Saver(max_to_keep=1)
        self.summ = []
        for k, v in self.log_vars:
            self.summ.append(tf.summary.scalar(k, v))

    def get_opt_reinit_op(self, opt, var_list, global_step):
        opt_slots = [opt.get_slot(var, name)
                    for name in opt.get_slot_names() for var in var_list]
        if isinstance(opt, tf.train.AdamOptimizer):
            opt_slots.extend([opt._beta1_power, opt._beta2_power])  # pylint: disable = W0212
        all_opt_variables = opt_slots + var_list + [global_step]
        opt_reinit_op = tf.variables_initializer(all_opt_variables)
        return opt_reinit_op

    def build_model_vaegan_test(self):

        self.x_lossy = arch.get_lossy(
            self.FLAGS, self.mdevice, self.images, self.theta_ph)
        self.z_batch = tf.Variable(tf.random_normal([self.batch_size, 128]), name='z_batch')
        self.x_p = self.M.generate(self.z_batch)
        self.x_p_lossy = arch.get_lossy(self.FLAGS, self.mdevice, self.x_p, self.theta_ph_xp)
        self.lp_lossy, logit = self.M.discriminate(self.x_p_lossy)
        self.lp, _ = self.M.discriminate(self.x_lossy, reuse=True)
        # define all losses
        m_loss1_batch = tf.reduce_mean((self.lp_lossy - self.lp)**2, (1, 2, 3))
        m_loss2_batch = tf.reduce_mean((self.x_lossy - self.x_p_lossy)**2, (1, 2, 3))
        zp_loss_batch = tf.reduce_sum(self.z_batch**2, 1)
        
        d_loss1_batch = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logit), logits=logit))
        d_loss2_batch = -1*tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logit), logits=logit))
        self.ml2_w, self.ml1_w, self.zp_w, self.dl1_w, self.dl2_w = self.FLAGS.ml2_w, self.FLAGS.ml1_w, self.FLAGS.zp_w, self.FLAGS.dl1_w, self.FLAGS.dl2_w
        # define total loss
        total_loss_batch = self.ml1_w * m_loss1_batch \
            + self.ml2_w * m_loss2_batch \
            + self.zp_w * zp_loss_batch \
            + self.dl1_w * d_loss1_batch \
            + self.dl2_w * d_loss2_batch
        self.total_loss = tf.reduce_mean(total_loss_batch)

        self.m_loss1 = tf.reduce_mean(m_loss1_batch)
        self.m_loss2 = tf.reduce_mean(m_loss2_batch)
        self.zp_loss = tf.reduce_mean(zp_loss_batch)
        self.d_loss1 = tf.reduce_mean(d_loss1_batch)
        self.d_loss2 = tf.reduce_mean(d_loss2_batch)


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
            self.encode_loss, var_list=self.e_vars + (self.g_vars if self.FLAGS.supervised else []) )
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
                    self.best_loss = np.load(self.ckp_dir + '/' + 'best_loss.npy')
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
            measure_dict = {
                'recon_loss':[],
                'psnr':[],
                'ssim':[]
            }
            
            # inf_net = inf_def.InferenceNetwork()
            
            while step <= self.max_iters:
                next_x_images = sess.run(self.next_x)

                theta_val = self.mdevice.sample_theta(
                    self.FLAGS, self.batch_size)
                theta_val_rec = self.mdevice.sample_theta(
                    self.FLAGS, self.batch_size)
                theta_val_xp = self.mdevice.sample_theta(
                    self.FLAGS, self.batch_size)
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
                    measure_dict['recon_loss'].append(rc)
                    # y_hat_val = inf_net.get_y_hat_val(rec_images)
                    # inception_score = get_inception_score(y_hat_val)
                    # score_list.append(inception_score)
                    # summary_str = sess.run(self.summ[0], feed_dict=fd_test)
                    # summary_str = sess.run(summary_op1)
                    if self.FLAGS.measurement_type == "blur_addnoise":
                        lossy_images = normalize(lossy_images)
                    if self.FLAGS.measurement_type == "drop_independent":
                        lossy_images = lossy_images * \
                            (1-self.FLAGS.drop_prob)
                    # summary_writer_test.add_summary(summary_str, step)
                    sample_images = [test_images[0:self.batch_size], lossy_images[0:self.batch_size], rec_images[0:self.batch_size],
                                     generated_image[0:self.batch_size]]
                    # save_images(sample_images[0:self.batch_size] , [self.batch_size/8, 8], '{}/train_{:02d}_recon.png'.format(self.sample_path, step))
                    titles = ['orig', 'lossy', 'reconstructed',
                              'generated']
                    save_images(sample_images, [self.batch_size/8, 8],
                                '{}/train_{:02d}_images.png'.format(self.log_dir, step), measure_dict, titles)
                if (step+1) % self.save_every == 0:
                    self.saver.save(sess, self.ckp_dir + '/last.ckpt',global_step=global_step, latest_filename='last')                                 
                    print("Model saved in file: %s" % self.ckp_dir)
                if (step+1)% (self.save_every//4) == 0:
                    if rc < self.best_loss:
                        self.best_loss = rc
                        np.save(self.ckp_dir + '/' + 'best_loss.npy', self.best_loss)
                        self.saver_best.save(sess, self.ckp_dir + '/best.ckpt',global_step=global_step, latest_filename='best')                                      
                        print("Best model saved in file: %s" % self.ckp_dir)

                step += 1
                new_learn_rate = sess.run(new_learning_rate)
                if new_learn_rate > 0.00005:
                    sess.run(add_global)

    def test(self, exp_name):
    
        if exp_name == 'iterative':
            # Set up gradient descent
            t_vars = tf.trainable_variables()
            g_vars = [var for var in t_vars if ('gen' in var.name) or ('dis' in var.name)]
            
            var_list = [self.z_batch]
            global_step = tf.Variable(0, trainable=False, name='global_step')
            learning_rate = tf.constant(self.FLAGS.lr_test)
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                opt = tf.train.AdamOptimizer(learning_rate)
                update_op = opt.minimize(
                    self.total_loss, var_list=var_list, global_step=global_step, name='update_op')

        self.saver = tf.train.Saver(
            var_list=g_vars) if exp_name == 'iterative' else tf.train.Saver()
        # self.opt_reinit_op = self.get_opt_reinit_op(opt, var_list, global_step)
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(self.val_init_op)
            # Initialzie the iterator
            sess.run(self.training_init_op)

            sess.run(init)
            
            if self.load_type != 'none' and os.path.exists(self.ckp_dir + '/' + self.load_type):
                ckpt = tf.train.get_checkpoint_state(
                    self.ckp_dir, latest_filename=self.load_type)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                    self.best_loss = np.load(self.ckp_dir + '/' + 'best_loss.npy')
                    print('model restored')
            test_images = sess.run(self.next_x_val)
            theta_val = self.mdevice.sample_theta(self.FLAGS, self.batch_size)
            # sess.run(self.opt_reinit_op)
            
            if exp_name == 'iterative':
                img2save= self.estimate(sess, learning_rate,
                            update_op, test_images, theta_val)
            elif exp_name == 'normal' or exp_name == 'supervised':
                feed_dict = {self.images: test_images,
                                self.theta_ph: theta_val}
                img2save = sess.run(self.x_tilde, feed_dict=feed_dict)
            else:
                img2save = self.get_unmeasure_pic(sess, test_images, theta_val)
            lossy = sess.run(self.x_lossy, feed_dict= {self.images: test_images, self.theta_ph: theta_val})
        lossy = np.clip(lossy, -1, 1)
        img2save = merge(img2save, [8,8])
        lossy = merge(lossy, [8, 8])
        test_images = merge(test_images, [8, 8])
        psnr, ssim = compute_pnsr_ssim(test_images, img2save)
        print ('psnr:{:.2f}, ssim:{:.2f}'.format(psnr, ssim))
        io.imsave('{}/{}.png'.format(self.log_dir,exp_name), img2save[:64])
        io.imsave('{}/orig.png'.format(self.log_dir), test_images[:64])
        io.imsave('{}/lossy.png'.format(self.log_dir), lossy[:64])


    def estimate(self ,sess, learning_rate, update_op,test_images, theta_val):
        measure_dict = {
            'recon_loss': [],
            'psnr': [],
            'ssim': []
        }
        for j in range(self.FLAGS.iter_test):
            theta_val_xp = self.mdevice.sample_theta(
                self.FLAGS, self.batch_size)
            feed_dict = {self.images: test_images,
                            self.theta_ph: theta_val, self.theta_ph_xp: theta_val_xp}
            _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val, \
                d_loss1_val, \
                d_loss2_val = sess.run([update_op, learning_rate, self.total_loss,
                                        self.m_loss1,
                                        self.m_loss2,
                                        self.zp_loss,
                                        self.d_loss1,
                                        self.d_loss2], feed_dict=feed_dict)
            logging_format = 'rr {} iter {} lr {:.3f} total_loss {:.3f} m_loss1 {:.3f} m_loss2 {:.3f} zp_loss {:.3f} d_loss1 {:.3f} d_loss2 {:.3f}'
            print(logging_format.format(1, j, lr_val, total_loss_val,
                                        m_loss1_val,
                                        m_loss2_val,
                                        zp_loss_val,
                                        d_loss1_val,
                                        d_loss2_val))
            if j % 10 == 0:
                titles = ['orig', 'lossy', 'reconstructed']
                images = sess.run(
                    [self.images, self.x_lossy, self.x_p], feed_dict=feed_dict)
                images[1] = np.clip(images[1], -1, 1)
                measure_dict['recon_loss'].append(
                    ((images[0] - images[2])**2).mean())
                save_images(images, [8, 8], '{}/test_{}_{}_{}/{}_images.png'.format(self.log_dir, self.ml1_w, self.dl1_w,
                                                                                                self.zp_w, j), measure_dict, titles)
        return images[2]

    def get_unmeasure_pic(self,sess, test_images, theta_val):

        measure_dict = {
                'recon_loss': [],
                'psnr': [],
                'ssim': []
        }
        fd = {self.images:test_images, self.theta_ph: theta_val}
        x_lossy = sess.run(self.x_lossy, feed_dict=fd)
        x_rec = self.mdevice.unmeasure_np(self.FLAGS, x_lossy , theta_val)
        x_rec = np.clip(x_rec, -1 , 1)
        
        images = merge(test_images,[8, 8])
        x_rec_merge = merge(x_rec, [8, 8])
        psnr, ssim = compute_pnsr_ssim(images, x_rec_merge)
        return x_rec


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

    
