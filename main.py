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

flags = tf.app.flags

flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("max_iters" , 600000, "the maxmization epoch")
flags.DEFINE_integer("latent_dim" , 128 , "the dim of latent code")
flags.DEFINE_float("lr" , 0.0003, "the init of learn rate")
#Please set this num of repeat by the size of your datasets.
flags.DEFINE_integer("repeat", 10000, "the numbers of repeat for your datasets")
flags.DEFINE_string("path", 'oem',
                    "for example, '/home/jack/data/' is the directory of your celebA data")
flags.DEFINE_integer("op", 0, "Training or Test")
flags.DEFINE_string('hparams','', 'Comma separated list of "name=value" pairs.')
flags.DEFINE_float('alpha',10, 'lambda1')
flags.DEFINE_float('beta',1, 'lambda2')
flags.DEFINE_integer('gamma', 0, 'recon or lossy recon')
flags.DEFINE_boolean('test', False, 'fast test')
flags.DEFINE_integer('print_every', 200, 'print every')
flags.DEFINE_integer('save_every', 2000, 'print every')
flags.DEFINE_string('load', 'none','load best last' )
flags.DEFINE_string('exp_name', '', 'name of experiment')
FLAGS = flags.FLAGS
hparams = hparams_def.get_hparams(FLAGS)
if __name__ == "__main__":
    FLAGS.path = '/home/' + FLAGS.path + '/.torch/data/'
    amb_utils.setup_vals(hparams)
    amb_utils.setup_dirs(hparams)
    exp_name = '/test_' if FLAGS.test else '/' + FLAGS.exp_name + '_'
    experiment = hparams.measurement_type + exp_name + "sample_lambda1_" + \
        str(FLAGS.alpha) + "_lambda2_" + \
        str(FLAGS.beta) + "_lambda3_" + str(FLAGS.gamma)
    # print and save hparams.pkl
    # basic_utils.print_hparams(hparams)
    # basic_utils.save_hparams(hparams)
    mdevice = measure.get_mdevice(hparams) 
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
    if not os.path.isdir(root_log_dir + '/test'):
        mkdir_p(root_log_dir + '/test')
    if not os.path.isdir(vaegan_checkpoint_dir):
        mkdir_p(vaegan_checkpoint_dir)

    os.system('fuser 6006/tcp -k')

    # model_path = None if FLAGS.load == 'none' else vaegan_checkpoint_dir +'/' + FLAGS.load + '.ckpt'  
    
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
                      ,hparams = hparams,_lambda = [FLAGS.alpha,FLAGS.beta, FLAGS.gamma], data_ob= cb_ob, print_every= print_every, save_every = save_every, ckp_dir = vaegan_checkpoint_dir)

    if FLAGS.op == 0:
        vaeGan.build_model_vaegan()
        vaeGan.train()

    else:
        vaeGan.build_model_vaegan_test()
        vaeGan.test()
