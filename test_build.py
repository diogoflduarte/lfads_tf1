import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
# suppresses logging of loading libcublas libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# suppress tf1 deprecation warnings
tf.logging.set_verbosity(tf.logging.ERROR)
from models import LFADS
from run_lfads_tf1 import FLAGS, build_hyperparameter_dict, hps_dict_to_obj, kind_dict
from lfads_wrapper.lfads_wrapper import lfadsWrapper
from data_funcs import load_datasets

data_dir = '/snel/home/lwimala/tmp/21_simons_workshop/lfads_input/'
model_dir = '/snel/home/lwimala/tmp/21_simons_workshop/lfads_output/'

hps_dict = build_hyperparameter_dict(FLAGS)

# set hyperparameter update dictionary
hp_update_dict = {
    # == set path to store model output
    'data_dir': data_dir,
    'data_filename_stem': 'lfads',
    'lfads_save_dir': model_dir,
    # == set architecture of LFADS model
    # ---- sets the dimensionality of RNNs, low-d factors, inputs to dynamial system 
    'gen_dim': 64, # generator dim
    'con_dim': 64, # controller dim
    'ic_enc_dim': 64, # initial condition encoder dim
    'ci_enc_dim': 64, # controller input encoder dim
    'factors_dim': 3, # factors dim (i.e., low-d factors)
    'co_dim': 1,      # controller output distribution dim (i.e., inputs "u")
    'output_dist': 'poisson', # noise emissions model to relate model output to data
    # == training hyperparameters
    # ---- controls batch size, learning rate init/annealing, early stopping
    'batch_size': 500, 
    'learning_rate_init': 1e-3, # initial learning rate
    'learning_rate_decay_factor': 0.95, # lr decay factor
    'learning_rate_n_to_compare': 6, # num. epochs before comparing lr
    'learning_rate_stop': 1e-5, 
    'target_num_epochs': 1, # tot. num. of epochs to train
    # == regularization hyperparameters
    # ---- controls dropout, coordinated dropout, l2/kl regularization weights/ramping
    'keep_prob': 0.95, # dropout keep prob (1-dropout_rate)
    'keep_ratio': 0.95, # coord. dropout keep prob (1 - cd_rate)
    'l2_gen_scale': 1e-3, # L2 reg. wt on generator recurrent kernel
    'l2_con_scale': 1e-3, # L2 reg. wt on controller recurrent kernel
    'kl_ic_weight': 1e-4, # KL wt on initial condition dist.
    'kl_co_weight': 1e-4, # KL wt on controller output dist.
    'kl_start_epoch': 0, # epoch to start applying KL
    'l2_start_epoch': 0, # epoch to start applying L2
    'kl_increase_epochs': 50, # num. epochs to linearly ramp KL wt
    'l2_increase_epochs': 50, # num. epochs to linearly ramp L2 wt
}

hps_dict.update(hp_update_dict)

for item in hps_dict.items():
    print(item)
hps = hps_dict_to_obj(hps_dict)

#datasets = load_datasets(data_dir, 'lfads', hps)

lfads = lfadsWrapper()
lfads.load_datasets_if_necessary(hps)

hps = lfads.infer_dataset_properties(hps)

tf.reset_default_graph()
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
allow_gpu_growth = True
config.gpu_options.allow_growth = allow_gpu_growth

sess = tf.Session(config=config)
try:
    with sess.as_default():
        with tf.device(hps.device):
            model = lfads.build_model(
                hps,
                datasets=lfads.datasets,
                reuse=tf.AUTO_REUSE)
except Exception:
    sess.close()
    tf.reset_default_graph()
    raise
