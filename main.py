import tensorflow as tf

from utils import mkdir_p
from vaegan import vaegan
from utils import CelebA
import shutil
import os
from commons import basic_utils
from commons import utils as amb_utils
from commons import arch
from commons import measure
from commons import hparams_def
import pdb
from options import FLAGS

if __name__ == "__main__":
    FLAGS.path = '/home/' + FLAGS.path + '/.torch/data/'
    amb_utils.setup_vals(FLAGS)
    # amb_utils.setup_dirs(FLAGS)
    exp_name = '/test_' if FLAGS.test else '/' + FLAGS.exp_name + '_'
    exp_name = exp_name + ('sup_' if FLAGS.supervised else '')
    experiment = FLAGS.measurement_type + exp_name + "sample_lambda1_" + \
        str(FLAGS.alpha) + "_lambda2_" + \
        str(FLAGS.beta) + "_lambda3_" + str(FLAGS.gamma)
    mdevice = measure.get_mdevice(FLAGS) 
    if FLAGS.dataset == 'mnist':
        root_log_dir = "./vaeganlogs_" + FLAGS.dataset + '/' + experiment
    else:
        root_log_dir = "./vaeganlogs/" + experiment
    vaegan_checkpoint_dir = "./model_vaegan/" + experiment 
    
    if FLAGS.load == 'none' and os.path.isdir(root_log_dir):
        print('Warning, you should set resume to non zero if you want to continue training')
        user_input = input('should we start from scratch?(y/n)')
        if not user_input.startswith('y'):
            exit(0)
        else:
            shutil.rmtree(root_log_dir)
            shutil.rmtree(vaegan_checkpoint_dir)
    if not os.path.isdir(root_log_dir):
        mkdir_p(root_log_dir)
    if not os.path.isdir(root_log_dir + '/test_{}_{}_{}'.format(FLAGS.ml1_w, FLAGS.dl1_w,
                                                              FLAGS.zp_w)):
        mkdir_p(root_log_dir + '/test_{}_{}_{}'.format(FLAGS.ml1_w, FLAGS.dl1_w,
                                                              FLAGS.zp_w))
    if not os.path.isdir(vaegan_checkpoint_dir):
        mkdir_p(vaegan_checkpoint_dir)

    os.system('fuser 6006/tcp -k')

    # model_path = None if FLAGS.load == 'none' else vaegan_checkpoint_dir +'/' + FLAGS.load + '.ckpt'  
    tf.set_random_seed(FLAGS.seed)
    batch_size = FLAGS.batch_size
    max_iters = FLAGS.max_iters if not FLAGS.test else 3
    latent_dim = FLAGS.latent_dim
    data_repeat = FLAGS.repeat
    print_every= FLAGS.print_every if not FLAGS.test else 1
    save_every = FLAGS.save_every
    learn_rate_init = FLAGS.lr
    cb_ob = CelebA(FLAGS.path)
    # import pdb; pdb.set_trace()
    vaeGan = vaegan(batch_size= batch_size, max_iters= max_iters, repeat = data_repeat,
                      load_type = FLAGS.load, latent_dim= latent_dim, log_dir= root_log_dir , learnrate_init= learn_rate_init , mdevice=mdevice
                     ,_lambda = [FLAGS.alpha,FLAGS.beta, FLAGS.gamma], data_ob= cb_ob, print_every= print_every, save_every = save_every, ckp_dir = vaegan_checkpoint_dir, flags = FLAGS)

    if FLAGS.op == 0:
        vaeGan.build_model_vaegan()
        vaeGan.train()

    else:
        if FLAGS.exp_name_test == 'iterative':
            vaeGan.build_model_vaegan_test()
        else:
            vaeGan.build_model_vaegan()
        vaeGan.test(FLAGS.exp_name_test)
