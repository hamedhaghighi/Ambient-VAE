import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("max_iters", 600000, "the maxmization epoch")
flags.DEFINE_integer("latent_dim", 128, "the dim of latent code")
flags.DEFINE_float("lr", 0.0003, "the init of learn rate")
#Please set this num of repeat by the size of your datasets.
flags.DEFINE_integer(
    "repeat", 10000, "the numbers of repeat for your datasets")
flags.DEFINE_string("path", 'oem',
                    "for example, '/home/jack/data/' is the directory of your celebA data")
flags.DEFINE_integer("op", 0, "Training or Test")
flags.DEFINE_integer("seed", 1, "Training or Test")
flags.DEFINE_string(
    'hparams', '', 'Comma separated list of "name=value" pairs.')
flags.DEFINE_float('alpha', 10, 'lambda1')
flags.DEFINE_float('beta', 1, 'lambda2')
flags.DEFINE_integer('gamma', 0, 'recon or lossy recon')
flags.DEFINE_boolean('test', False, 'fast test')
flags.DEFINE_boolean('supervised', False, 'fast test')
flags.DEFINE_boolean('blur', False, 'fast test')
flags.DEFINE_integer('print_every', 200, 'print every')
flags.DEFINE_integer('save_every', 2000, 'print every')
flags.DEFINE_string('load', 'none', 'load best last')
flags.DEFINE_string('measurement_type', 'drop_patch', 'name of experiment')
flags.DEFINE_string('dataset', 'celebA', 'name of experiment')
flags.DEFINE_string('exp_name', '', 'name of experiment')
flags.DEFINE_string('exp_name_test', 'normal', 'name of experiment')
flags.DEFINE_string('unmeasure_type', 'blur', 'name of experiment')
flags.DEFINE_string('train_mode', 'ambient', 'name of experiment')
flags.DEFINE_float("lr_test", 0.1, "the init of learn rate")
flags.DEFINE_float("x_min", -1, "the init of learn rate")
flags.DEFINE_float("x_max", 1, "the init of learn rate")
flags.DEFINE_float("l2_w", 1e-6, "the init of learn rate")
flags.DEFINE_float("ml1_w", 5, "the init of learn rate")
flags.DEFINE_float("ml2_w", 0, "the init of learn rate")
flags.DEFINE_float("dl1_w", 0, "the init of learn rate")
flags.DEFINE_float("dl2_w", 0, "the init of learn rate")
flags.DEFINE_float("zp_w", 0.001, "the init of learn rate")
flags.DEFINE_float("drop_prob", 0.9, "the init of learn rate")
flags.DEFINE_float("blur_radius", 1, "the init of learn rate")
flags.DEFINE_float("additive_noise_std", 0.2, "the init of learn rate")
flags.DEFINE_float("signal_power", 0.2885201, "the init of learn rate")
flags.DEFINE_integer("iter_test", 100, "the init of learn rate")
flags.DEFINE_integer("num_angles", 1, "the init of learn rate")
flags.DEFINE_integer("patch_size", 32, "the init of learn rate")
flags.DEFINE_integer("blur_filter_size", 5, "the init of learn rate")
flags.DEFINE_integer("c_dim", 3, "the init of learn rate")
flags.DEFINE_list("image_dims", [64,64,3], "the init of learn rate")

FLAGS = flags.FLAGS
