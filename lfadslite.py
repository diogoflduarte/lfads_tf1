from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import time
import os
import re
import logging
import json
import shlex
#import random
#import matplotlib.pyplot as plt
import subprocess

from helper_funcs import linear, makeInitialState
from helper_funcs import kind_dict, kind_dict_key
from helper_funcs import DiagonalGaussianFromInput, DiagonalGaussian
from helper_funcs import DiagonalGaussianFromExisting, LearnableDiagonalGaussian, diag_gaussian_log_likelihood
from helper_funcs import LinearTimeVarying
from helper_funcs import KLCost_GaussianGaussian
from helper_funcs import write_data
from helper_funcs import printer, mkdir_p
#from plot_funcs import plot_data, close_all_plots
#from data_funcs import read_datasets
from customcells import ComplexCell
from rnn_helper_funcs import BidirectionalDynamicRNN, DynamicRNN
from helper_funcs import dropout
# TPU related imports
#from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
#from tensorflow.contrib.cluster_resolver import TPUClusterResolver



# this will be used to store matrices/vectors for use in tf.case statements
def makelambda(v):          # Used with tf.case
    return lambda: v

# this is used to setup a selector that is session-specific
# it's a wrapper around tf.case, ensures there is no default (returns error if default is reached)
def _case_with_no_default(pairs):
    def _default_value_fn():
        with tf.control_dependencies([tf.Assert(False, ["Reached default"])]):
            return tf.identity(pairs[0][1]())
    return tf.case(pairs, _default_value_fn, exclusive=True)

class Logger(object):
    def __init__(self, log_file):
        self.logfile = log_file
        if not os.path.exists(log_file):
            open(log_file, 'w').close()
    def printlog(self, *args):
        strtext = (('{} ' * len(args)).format(*args))[:-1]
        #self.logfile.write(strtext)
        with open(self.logfile, 'a') as f:
            print(strtext, file=f)
            print(strtext)
            
# Custom logging filter
class CustomFilter(logging.Filter):
    """ pass the messages that contain "total_cost" only """
    def filter(self, record):
        isCost = 'total' in record.getMessage()
        return isCost


class CustomFormatter(logging.Formatter):
    def parse_message(self, msg):
        msg = "".join(msg.split())
        # separate the costs from the (time taken)
        msg = msg.split('(')
        tmp_msg = msg[0].split(':')
        msg[0] = tmp_msg[-1]
        val_dict = dict(x.split('=') for x in msg[0].split(','))
        val_dict['time'] = msg[1].replace('sec)', '') if len(msg) > 1 else 'NA'
        return val_dict

    def format(self, record):
        res = super(CustomFormatter, self).format(record)
        value_dict = self.parse_message(res)

        if float(value_dict.get('eval_data_mode', 0)) == 1.:
            # evaluation on training data
            tr_total, val_total = (value_dict.get('eval_total', 'NA'), 'NA')
            tr_heldin, tr_heldout, val_heldin, val_heldout = (value_dict.get('eval_recon_heldin_samps', 'NA'),
                                                              value_dict.get('eval_recon_heldout_samps', 'NA'),
                                                              'NA', 'NA')
        else:
            # evaluation on validation data
            # or during training
            tr_total, val_total = (value_dict.get('train_total', 'NA'), value_dict.get('eval_total', 'NA'))
            tr_heldin, tr_heldout, val_heldin, val_heldout = (value_dict.get('train_recon_heldin_samps', 'NA'),
                                                              value_dict.get('train_recon_heldout_samps', 'NA'),
                                                              value_dict.get('eval_recon_heldin_samps', 'NA'),
                                                              value_dict.get('eval_recon_heldout_samps', 'NA'))

        formatted_str = 'epoch, {}, step, {}, total, {}, {}, recon, {}, {}, {}, {}, time, {}'.format(
            'NA', value_dict.get('step', 'NA'), tr_total, val_total,
            tr_heldin, tr_heldout, val_heldin, val_heldout, value_dict.get('time', 'NA')
        )
        return formatted_str

class TrainHook(tf.train.SessionRunHook):
    def __init__(self, lr):
        self.lr = lr
    def before_run(self, run_context):
        if self.lr is not None:
            return tf.train.SessionRunArgs(self.lr.initializer)

# todo: add decay learning rate
# todo: add early stopping
# todo: add posertior mean sampling
# todo: add support for multi-session
# todo, now resest learning rate on every call to session.run (fine for PBT but need to change it for normal run)
# todo, allow setting checkpoint name
# todo, saveing/loading lve checkpoint
# todo, pass the RunConfig setting as params to train

class LFADS(object):
    def __init__(self, hps, datasets):
        # store the hps
        print(hps)
        hps['n_epochs_eval'] = 1
        self.datasets = datasets
        #self.setup_stdout_logger(hps)
        if hps['device'].lower() == 'tpu':
            hps['use_tpu'] = True
        else:
            hps['use_tpu'] = False

        self.use_tpu = hps['use_tpu']

        if self.use_tpu:
            my_project = subprocess.check_output([
                'gcloud','config','get-value','project'])
            my_zone = subprocess.check_output([
                'gcloud','config','get-value','compute/zone'])
            print(my_project, my_zone)
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                    tpu=[os.environ['TPU_NAME']],
                    #zone=my_zone,
                    #project=my_project
                    )
            self.tpu_cluster_resolver = tpu_cluster_resolver
            #self.master = tpu_cluster_resolver.get_master()
        else:
            self.tpu_cluster_resolver = None

    def _data_fn(self, mode, num_epochs=None):
        def input_fn(params):
            # constructing Datasets using tf.data
            datasets = self.datasets
            batch_size = params['batch_size']

            for name, data_dict in datasets.items():
                if mode == tf.estimator.ModeKeys.TRAIN:
                    data_mode = np.ones([data_dict['train_data'].shape[0],1], dtype=np.float32)
                    print(name)
                    name = np.repeat(np.array(name, dtype=np.str), data_dict['train_data'].shape[0])
                    print(name.shape)
                    in_data = {'data':data_dict['train_data'].astype(np.float32),
                               'data_name':name,
                               'data_mode':data_mode}

                    return tf.estimator.inputs.numpy_input_fn(in_data,
                                                              batch_size=batch_size,
                                                              num_epochs=num_epochs,
                                                              shuffle=True,
                                                              num_threads=1)
                else:
                    data_mode = np.ones([data_dict['valid_data'].shape[0],0], dtype=np.float32)
                    name = np.repeat(np.array(name, dtype=np.str), data_dict['valid_data'].shape[0])
                    in_data = {'data':data_dict['valid_data'].astype(np.float32),
                               'data_name':name,
                               'data_mode':data_mode}
                    return tf.estimator.inputs.numpy_input_fn(in_data,
                                                              batch_size=batch_size,
                                                              num_epochs=1,
                                                              shuffle=False,
                                                              num_threads=1)
        return input_fn

    def data_fn(self, mode, data_type):

      def input_fn(params):
        # TODO: adapt this for multiple sessions
        # constructing Datasets using tf.data
        batch_size = params['batch_size']
        # datasets are already loaded at init
        datasets = self.datasets

        dataset = {}
        for name, data_dict in datasets.items():
            # reading the data from numpy array and preparing tf.Dataset
            if data_type == 'train':
                data = data_dict['train_data']
                data_cv_mask = data_dict['train_data_cvmask']
                data_ext_input = data_dict['train_ext_input']
                data_truth = data_dict['train_truth']
            else:
                data = data_dict['valid_data']
                data_cv_mask = data_dict['valid_data_cvmask']
                data_ext_input = data_dict['valid_ext_input']
                data_truth = data_dict['valid_truth']
            
            # preparing the dict to pass to tf.Dataset
            data_d = {'data':data.astype(np.float32)}
            if data_cv_mask is not None:
                data_d['data_cv_mask'] = data_cv_mask.astype(np.float32)
            if data_ext_input is not None:
                data_d['data_ext_input'] = data_ext_input.astype(np.float32)
            if data_truth is not None:
                data_d['data_truth'] = data_truth.astype(np.float32)

            #data_mode = np.ones([data.shape[0],1], dtype=np.float32)
            #data_d['data_mode']:data_mode
            # the index of the dataset for multi-session case
            data_name = np.repeat(np.array(0, dtype=np.float32), data.shape[0])
            data_d['data_name'] = data_name

            dataset[name] = tf.data.Dataset.from_tensor_slices(data_d)
            dataset[name] = dataset[name].cache()
        
            if mode == tf.estimator.ModeKeys.TRAIN:
                # shuffle when training
                dataset[name] = dataset[name].shuffle(buffer_size=batch_size*5)

            dataset[name] = dataset[name].repeat()
            dataset[name].prefetch(batch_size)
            dataset[name] = dataset[name].apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        self.dname = name
        return dataset[name]
      
      return input_fn

    def lfads_model_fn(self, features, mode, params):
        
        # to stop certain gradients paths through the graph in backprop
        def entry_stop_gradients(target, mask):
            mask_h = tf.abs(1. - mask)
            return tf.stop_gradient(mask_h * target) + mask * target

        # type of GRU cell used (only for co_dim=0)
        CELL_TYPE = 'customgru'
        hps = params
        # pull out some parameters
        batch_size = hps['batch_size']
        # no learning rate decay applied
        learning_rate = hps['learning_rate_init'] 
        cv_keep_ratio = hps['cv_keep_ratio']
        cd_grad_passthru_prob = hps['cd_grad_passthru_prob']

        # alias for names
        dataset_ph = features['data']
        # external input if exists
        ext_input = features['data_ext_input'] if hps['ext_input_dim'] > 0 else None
        # cv_mask if exists
        cv_binary_mask = features['data_cv_mask'] if hps['cv_keep_ratio'] < 1 else tf.ones_like(dataset_ph)
        # truth data if exists
        data_truth = features['data_truth'] if 'data_truth' in features else None
        
        #dataName = tf.convert_to_tensor(self.dname)#features['data_name']


        run_type = kind_dict("train")

        # only when training do
        if mode == tf.estimator.ModeKeys.TRAIN:
            keep_prob = tf.convert_to_tensor(hps['keep_prob'])
            keep_ratio = tf.convert_to_tensor(hps['keep_ratio'])
        else:
            keep_prob = tf.convert_to_tensor(1.0)
            keep_ratio = tf.convert_to_tensor(1.0)

        kl_ic_weight = tf.convert_to_tensor(hps['kl_ic_weight'])
        kl_co_weight = tf.convert_to_tensor(hps['kl_co_weight'])

        # make placeholders for all the input and output adapter matrices
        ndatasets = hps['ndatasets']
        # preds will be used to select elements of each session

        preds = [None] * ndatasets
        fns_in_fac_Ws = [None] * ndatasets
        fns_in_fac_bs = [None] * ndatasets
        fns_out_fac_Ws = [None] * ndatasets
        fns_out_fac_bs = [None] * ndatasets
        dataset_names = hps['dataset_names']
        # specific to lfadslite - need to make placeholders for the cross validation dropout masks
        random_masks_np = [None] * ndatasets
        fns_cv_binary_masks = [None] * ndatasets
        # ext_inputs are not yet implemented, but, placeholder for later

        # figure out the input (dataset) dimensionality
        # allsets = hps['dataset_dims'].keys()
        # input_dim = hps['dataset_dims'][allsets[0]]

        ## do per-session stuff
        # for d, name in enumerate(dataset_names):
        #     data_dim = hps.dataset_dims[name]
        #     # Step 0) define the preds comparator for this dataset
        #     preds[d] = tf.equal(tf.constant(name), dataName)
        #     # Step 1) alignment matrix stuff.
        #     # the alignment matrix only matters if in_factors_dim is nonzero
        #     in_mat_cxf = None
        #     align_bias_1xc = None
        #     in_bias_1xf = None
        #     if hps.in_factors_dim > 0:
        #         # get the alignment_matrix if provided
        #         if 'alignment_matrix_cxf' in datasets[name].keys():
        #             in_mat_cxf = datasets[name]['alignment_matrix_cxf'].astype(np.float32)
        #             # check that sizing is appropriate
        #             if in_mat_cxf.shape != (data_dim, hps.in_factors_dim):
        #                 raise ValueError("""Alignment matrix must have dimensions %d x %d
        #                  (data_dim x factors_dim), but currently has %d x %d.""" %
        #                                  (data_dim, hps.in_factors_dim, in_mat_cxf.shape[0],
        #                                   in_mat_cxf.shape[1]))
        #         if 'alignment_bias_c' in datasets[name].keys():
        #             align_bias_c = datasets[name]['alignment_bias_c'].astype(np.float32)
        #             align_bias_1xc = np.expand_dims(align_bias_c, axis=0)
        #             if align_bias_1xc.shape[1] != data_dim:
        #                 raise ValueError("""Alignment bias must have dimensions %d
        #                  (data_dim), but currently has %d.""" %
        #                                  (data_dim, in_mat_cxf.shape[0]))
        #         if in_mat_cxf is not None and align_bias_1xc is not None:
        #             # (data - alignment_bias) * W_in
        #             # data * W_in - alignment_bias * W_in
        #             # So b = -alignment_bias * W_in to accommodate PCA style offset.
        #             in_bias_1xf = -np.dot(align_bias_1xc, in_mat_cxf)
        #         # initialize a linear transform based on the above
        #         in_fac_linear = init_linear_transform(data_dim, hps.in_factors_dim, mat_init_value=in_mat_cxf,
        #                                               bias_init_value=in_bias_1xf,
        #                                               name=name + '_in_fac_linear')
        #         in_fac_W, in_fac_b = in_fac_linear
        #         # to store per-session matrices/biases for later use, need to use 'makelambda'
        #         fns_in_fac_Ws[d] = makelambda(in_fac_W)
        #         fns_in_fac_bs[d] = makelambda(in_fac_b)

        #     # single-sample cross-validation mask
        #     # generate one random mask once (for each dataset) when building the graph
        #     # use a different (but deterministic) random seed for each dataset (hence adding 'd' below)
        #     if hps.cv_rand_seed:
        #         np.random.seed(int(hps.cv_rand_seed) + d)

        #     # Step 2) make a cross-validation random mask for each dataset
        #     # uniform [cv_keep_ratio, 1.0 + cv_keep_ratio)
        #     random_mask_np_this_dataset = cv_keep_ratio + np.random.rand(hps['num_steps'], hps.dataset_dims[name])
        #     # convert to 0 and 1
        #     random_mask_np_this_dataset = np.floor(random_mask_np_this_dataset)
        #     random_masks_np[d] = random_mask_np_this_dataset
        #     # converting to tensor
        #     fns_cv_binary_masks[d] = makelambda(tf.convert_to_tensor(random_masks_np[d], dtype=tf.float32))

        #     # reset the np random seed to enforce randomness for the other random draws
        #     np.random.seed()

        #     # Step 3) output matrix stuff
        #     out_mat_fxc = None
        #     out_bias_1xc = None

        #     # if input and output factors dims match, can initialize output matrices using transpose of input matrices
        #     if in_mat_cxf is not None:
        #         if hps.in_factors_dim == hps.factors_dim:
        #             out_mat_fxc = in_mat_cxf.T
        #     if align_bias_1xc is not None:
        #         out_bias_1xc = align_bias_1xc

        #     if hps.output_dist.lower() == 'poisson':
        #         output_size = data_dim
        #     elif hps.output_dist.lower() == 'gaussian':
        #         output_size = data_dim * 2
        #         if out_mat_fxc is not None:
        #             out_mat_fxc = tf.concat([out_mat_fxc, out_mat_fxc], 0)
        #         if out_bias_1xc is not None:
        #             out_bias_1xc = tf.concat([out_bias_1xc, out_bias_1xc], 0)
        #     out_fac_linear = init_linear_transform(hps.factors_dim, output_size, mat_init_value=out_mat_fxc,
        #                                            bias_init_value=out_bias_1xc,
        #                                            name=name + '_out_fac_linear')
        #     out_fac_W, out_fac_b = out_fac_linear
        #     fns_out_fac_Ws[d] = makelambda(out_fac_W)
        #     fns_out_fac_bs[d] = makelambda(out_fac_b)

        # # now 'zip' together the 'pred' selector with all the function handles
        # pf_pairs_in_fac_Ws = zip(preds, fns_in_fac_Ws)
        # pf_pairs_in_fac_bs = zip(preds, fns_in_fac_bs)
        # pf_pairs_out_fac_Ws = zip(preds, fns_out_fac_Ws)
        # pf_pairs_out_fac_bs = zip(preds, fns_out_fac_bs)
        # pf_pairs_cv_binary_masks = zip(preds, fns_cv_binary_masks)

        # # now, choose the ones for this session
        # if hps.in_factors_dim > 0:
        #     this_dataset_in_fac_W = _case_with_no_default(pf_pairs_in_fac_Ws)
        #     this_dataset_in_fac_b = _case_with_no_default(pf_pairs_in_fac_bs)

        # this_dataset_out_fac_W = _case_with_no_default(pf_pairs_out_fac_Ws)
        # this_dataset_out_fac_b = _case_with_no_default(pf_pairs_out_fac_bs)
        # this_dataset_cv_binary_mask = _case_with_no_default(pf_pairs_cv_binary_masks)

        # batch_size - read from the data placeholder
         #tf.shape(dataset_ph)[0]
        # batch_size = hps['batch_size']
        # batch_size = dataset_ph.get_shape().as_list()[0]

        # can we infer the data dimensionality for the random mask?
        # data_dim_tensor = tf.shape(dataset_ph)[2]
        # dataset_dims = dataset_ph.get_shape()
        # data_dim = dataset_dims[2]
        # self.printlog(data_dim)

        # coordinated dropout
        # apply dropout to the data
        dataset_ph = tf.nn.dropout(dataset_ph, keep_prob)
        # batch_size - read from the data placeholder

        # MRK, seq_len for dynamicrnn (must be numpy array not tensor)
        seq_len = hps['num_steps'] * np.ones([batch_size], dtype=np.int32)

        # coordinated dropout
        if hps['keep_ratio'] < 1:
            # coordinated dropout enabled on inputs
            masked_dataset_ph, coor_drop_binary_mask = dropout(dataset_ph, keep_ratio)
        else:
            # no coordinated dropout
            masked_dataset_ph = dataset_ph

        # apply cross-validation dropout
        masked_dataset_ph = tf.div(masked_dataset_ph, cv_keep_ratio) * cv_binary_mask

        # define input to encoders
        if hps['in_factors_dim'] == 0:
           input_to_encoders = masked_dataset_ph
        else:
            input_factors_object = LinearTimeVarying(inputs = masked_dataset_ph,
                                                    output_size = hps['in_factors_dim'],
                                                    transform_name = 'data_2_infactors',
                                                    output_name = 'infactors',
                                                    W = this_dataset_in_fac_W,
                                                    b = this_dataset_in_fac_b,
                                                    nonlinearity = None)
            input_to_encoders = input_factors_object.output
            
        with tf.variable_scope('ic_enc'):
            #ic_enc_cell = CustomGRUCell(num_units = hps['ic_enc_dim'],\
            #                            batch_size = batch_size,
            #                            clip_value = hps['cell_clip_value'],
            #                            recurrent_collections=['l2_ic_enc'])
            
            #seq_len.set_shape([1, batch_size])
            ## ic_encoder
            ic_enc_rnn_obj = BidirectionalDynamicRNN(
                state_dim = hps['ic_enc_dim'],
                batch_size = batch_size,
                name = 'ic_enc',
                sequence_length = seq_len,
                inputs = input_to_encoders,
                initial_state = None,
                clip_value = hps['cell_clip_value'],
                recurrent_collections='l2_ic_enc',
                rnn_type = CELL_TYPE)
            #    cell = ic_enc_cell
            #    output_keep_prob = batch_sizekeep_prob

            # wrap the last state with a dropout layer
            ic_enc_laststate_dropped = tf.nn.dropout(ic_enc_rnn_obj.last_tot, keep_prob)
            
            # map the ic_encoder onto the actual ic layer
            gen_ics_posterior = DiagonalGaussianFromInput(
                x = ic_enc_laststate_dropped,
                z_size = hps['ic_dim'],
                name = 'ic_enc_2_ics',
                var_min = hps['ic_post_var_min'],
            )

        # to go forward, either sample from the posterior, or take mean
        if run_type in [kind_dict("train"), kind_dict("posterior_sample_and_average")]:
            gen_ics_lowd = gen_ics_posterior.sample()
        else:
            gen_ics_lowd = gen_ics_posterior.mean
        
        with tf.variable_scope('generator'):
            # lstms have twice the number of state dims as everybody else (h and c cells) - correct for that here.
            if CELL_TYPE.lower() == 'lstm':
                gen_ics = linear(gen_ics_lowd, hps['gen_dim']*2, name='ics_2_g0')
            else:
                gen_ics = linear(gen_ics_lowd, hps['gen_dim'], name='ics_2_g0')


        ### CONTROLLER construction
        # this should only be done if a controller is requested
        # if not, skip all these graph elements like so:
        print('TEST1', input_to_encoders.get_shape())
        if hps['co_dim'] == 0:
            with tf.variable_scope('generator'):
                # setup generator
                # ext_input will be None if not provided
                gen_input = ext_input

                gen_rnn_obj = DynamicRNN(state_dim = hps['gen_dim'],
                                              batch_size = batch_size,
                                              name = 'gen',
                                              sequence_length=seq_len,
                                              inputs = gen_input,
                                              initial_state = gen_ics,
                                              rnn_type = CELL_TYPE,
                                              recurrent_collections='l2_gen',
                                              clip_value = hps['cell_clip_value']
                )
                
                gen_states = gen_rnn_obj.states

            with tf.variable_scope('factors'):
                # wrap the generator states in a dropout layer
                gen_states_dropped = tf.nn.dropout(gen_rnn_obj.states, keep_prob)
                ## factors
                fac_obj = LinearTimeVarying(inputs=gen_states_dropped,
                                                 output_size=hps['factors_dim'],
                                                 transform_name='gen_2_factors',
                                                 output_name='factors_concat',
                                                 collections='l2_gen_2_factors',
                                                 do_bias=False,
                                                 normalized=True
                )
                factors = fac_obj.output
        else:
            # co_dim > 0 case: when we have controller
            with tf.variable_scope('ci_enc'):
                # construct controller input encoder
                ci_enc_rnn_obj = BidirectionalDynamicRNN(
                    state_dim = hps['ci_enc_dim'],
                    batch_size = batch_size,
                    name = 'ci_enc',
                    sequence_length=seq_len,
                    inputs = input_to_encoders,
                    initial_state = None,
                    rnn_type = CELL_TYPE,
                    recurrent_collections='l2_ci_enc',
                    clip_value = hps['cell_clip_value'])
                
                if not hps['do_causal_controller']:
                    ci_enc_rnn_states = ci_enc_rnn_obj.states
                else:
                    # if causal controller, only use the fwd rnn
                    [ci_enc_fwd_states, _]  = ci_enc_rnn_obj.states
                    if hps['controller_input_lag'] > 0:
                        toffset = hps['controller_input_lag']
                        leading_zeros = tf.zeros_like(ci_enc_fwd_states[:,0:toffset,:])
                        ci_enc_rnn_states = tf.concat([leading_zeros,
                                                            ci_enc_fwd_states[:,0:-toffset,:]],
                                                           axis=1)
                    else:
                        [ci_enc_rnn_states, _]  = ci_enc_rnn_obj.states
                
                # states from bidirec RNN come out as a tuple. concatenate those:
                ci_enc_rnn_states = tf.concat(ci_enc_rnn_states,
                                                   axis=2, name='ci_enc_rnn_states')

                ## take a linear transform of the ci_enc output
                #    this is to lower the dimensionality of the ci_enc
                #with tf.variable_scope('ci_enc_2_co_in'):
                #    # one input from the encoder, another will come back from factors
                #    ci_enc_object = LinearTimeVarying(
                #        inputs=ci_enc_rnn_states,
                #        output_size = hps['con_ci_enc_in_dim'],
                #        transform_name = 'ci_enc_2_co_in',
                #       output_name = 'ci_enc_output_concat',
                #        nonlinearity = None,
                #        collections='l2_ci_enc_2_co_in')
                #    ci_enc_outputs = ci_enc_object.output

                # pass ci_enc output directly to the controller
                ci_enc_outputs = ci_enc_rnn_states
            ## the controller, controller outputs, generator, and factors are implemented
            #     in one RNN whose individual cell is "complex"
            #  this is required do to feedback pathway from factors->controller.
            #    impossible to dynamically unroll with separate RNNs.
            with tf.variable_scope('complexcell'):
                # the "complexcell" architecture requires an initial state definition
                # have to define states for each of the components, then concatenate them
                con_init_state = makeInitialState(hps['con_dim'],
                                                  batch_size,
                                                  'controller')
                co_mean_init_state = makeInitialState(hps['co_dim'],
                                                      batch_size,
                                                      'controller_output_mean')
                co_logvar_init_state = makeInitialState(hps['co_dim'],
                                                        batch_size,
                                                        'controller_output_logvar')
                co_sample_init_state = makeInitialState(hps['co_dim'],
                                                        batch_size,
                                                        'controller_output_sample')
                fac_init_state = makeInitialState(hps['factors_dim'],
                                                  batch_size,
                                                  'factor')
                comcell_init_state = [con_init_state, gen_ics,
                                           co_mean_init_state, co_logvar_init_state,
                                           co_sample_init_state, fac_init_state]
                complexcell_init_state = tf.concat(axis=1, values = comcell_init_state)


                # here is what the state vector will look like
                comcell_state_dims = [hps['gen_dim'],
                                      hps['con_dim'],
                                      hps['co_dim'], # for the controller output means
                                      hps['co_dim'], # for the variances
                                      hps['co_dim'], # for the sampled controller output
                                      hps['factors_dim']]


                # construct the complexcell
                complexcell=ComplexCell(num_units_gen=hps['gen_dim'],
                                             num_units_con=hps['con_dim'],
                                             factors_dim=hps['factors_dim'],
                                             co_dim=hps['co_dim'],
                                             ext_input_dim=hps['ext_input_dim'],
                                             inject_ext_input_to_gen=True,
                                             var_min=hps['co_post_var_min'],
                                             run_type=run_type,
                                             keep_prob=keep_prob)

                # construct the actual RNN
                #   its inputs are the output of the controller_input_enc

                if hps['ext_input_dim']:
                    complex_cell_inputs = tf.concat(axis=2, values = [ci_enc_outputs, ext_input])
                else:
                    complex_cell_inputs = ci_enc_outputs
                complex_outputs, complex_final_state =\
                tf.nn.dynamic_rnn(complexcell,
                                  inputs = complex_cell_inputs,
                                  sequence_length=seq_len,
                                  initial_state=complexcell_init_state)

                # split the states of the individual RNNs
                # from the packed "complexcell" state
                gen_states, con_states, co_mean_states, co_logvar_states, controller_outputs, factors =\
                tf.split(complex_outputs, comcell_state_dims, axis=2)

        # now back to code that runs for all models
        with tf.variable_scope('rates'):
            ## "rates" - more properly called "output_distribution"
            if hps['output_dist'].lower() == 'poisson':
                nonlin = 'exp'
            elif hps['output_dist'].lower() == 'gaussian':
                nonlin = None
            else:
                raise NameError("Unknown output distribution: " + hps['output_dist'])
                
            # rates are taken as a linear (or nonlinear) readout from the factors
            rates_object = LinearTimeVarying(inputs = factors,
                                             #output_size = output_size,
                                             output_size=dataset_ph.get_shape()[2],
                                             transform_name = 'factors_2_rates',
                                             output_name = 'rates_concat',
                                             #W = this_dataset_out_fac_W,
                                             #b = this_dataset_out_fac_b,
                                             nonlinearity = nonlin)

            # select the relevant part of the output depending on model type
            if hps['output_dist'].lower() == 'poisson':
                # get both the pre-exponentiated and exponentiated versions
                logrates=rates_object.output
                output_dist_params=rates_object.output_nl
            elif hps['output_dist'].lower() == 'gaussian':
                # get linear outputs, split into mean and variance
                output_mean, output_logvar = tf.split(rates_object.output, 2, axis=2)
                output_dist_params=rates_object.output

        # if we are running the model for prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'gen_ics': gen_ics,
                'gen_states': gen_states,
                'factors': factors,
                'output_dist_params': output_dist_params
            }
            #export_outputs = {
            #    'predict': tf.estimator.export.PredictOutput(predictions)
            #}
            return tpu_estimator.TPUEstimatorSpec(mode, 
                                              predictions=predictions,
                                              #export_outputs=export_outputs
                                              )

       ## calculate the KL cost
        # g0 - build a prior distribution to compare to
        gen_ics_prior = LearnableDiagonalGaussian(
            batch_size=batch_size,
            z_size = [hps['ic_dim']], name='gen_ics_prior',
            var = hps['ic_prior_var'])

        kl_cost = tf.constant(0, dtype=tf.float32)
        if hps['kl_ic_weight'] > 0:
            # g0 KL cost for each trial
            kl_cost_g0_b = KLCost_GaussianGaussian(gen_ics_posterior,
                                                        gen_ics_prior).kl_cost_b
            # g0 KL cost for the whole batch
            kl_cost_g0 = tf.reduce_mean(kl_cost_g0_b)
            # total KL cost
            kl_cost = kl_cost_g0 * kl_ic_weight

        if hps['co_dim'] > 0:
            # if there are controller outputs, calculate a KL cost for them
            # first build a prior to compare to
            # Controller outputs
            #autocorrelation_taus = [hps.prior_ar_atau for x in range(hps.co_dim)]
            #noise_variances = [hps.prior_ar_nvar for x in range(hps.co_dim)]
            #cos_prior = prior_zs_ar_con = \
            #    LearnableAutoRegressive1Prior(batch_size, hps.co_dim,
            #                                  autocorrelation_taus,
            #                                  noise_variances,
            #                                  hps.do_train_prior_ar_atau,
            #                                  hps.do_train_prior_ar_nvar,
            #                                  hps['num_steps'], "u_prior_ar1")

            # zero mean DiagonalGaussian
            cos_prior = DiagonalGaussian(
                z_size = [batch_size, hps['num_steps'], hps['co_dim']],
                name='cos_prior', var = hps['co_prior_var'])

            # then build a posterior
            cos_posterior = DiagonalGaussianFromExisting(co_mean_states, co_logvar_states)

            #print(co_mean_states)
            # MRK, changed this to LFADS'
            #kl_cost_co_b_t = \
            #    KLCost_GaussianGaussianProcessSampled(
            #        cos_posterior, cos_prior).kl_cost_b

            if hps['kl_co_weight'] > 0:
                # CO KL cost per timestep
                kl_cost_co_b_t = KLCost_GaussianGaussian(cos_posterior, cos_prior).kl_cost_b
                # CO KL cost for the batch
                kl_cost_co = tf.reduce_mean(
                    tf.reduce_mean(kl_cost_co_b_t, [1]) )
                kl_cost += kl_cost_co * kl_co_weight

        ## calculate reconstruction cost
        # get final mask for gradient blocking
        if hps['keep_ratio'] < 1:
            # let the gradients pass through on blocked nodes with some probability
            random_tensor = tf.convert_to_tensor(1. - cd_grad_passthru_prob)
            random_tensor += tf.random_uniform(tf.shape(coor_drop_binary_mask),
                                                       dtype=coor_drop_binary_mask.dtype)
            # pass thru some gradients
            tmp_binary_mask =  coor_drop_binary_mask * tf.floor(random_tensor)
            # exclude cv samples
            grad_binary_mask = cv_binary_mask * (1. - tmp_binary_mask)
        else:
            grad_binary_mask = cv_binary_mask

        print('Test2', logrates.get_shape())
        # block gradients for coordinated dropout and cross-validation
        if hps['output_dist'].lower() == 'poisson':
            masked_logrates = entry_stop_gradients(logrates, grad_binary_mask)
            #loglikelihood_b_t = Poisson(masked_logrates).logp(dataset_ph)
            loglikelihood_b_t = -tf.nn.log_poisson_loss( dataset_ph, masked_logrates, compute_full_loss=True )
        elif hps['output_dist'].lower() == 'gaussian':
            masked_output_mean = entry_stop_gradients(output_mean, grad_binary_mask)
            masked_output_logvar = entry_stop_gradients(output_logvar, grad_binary_mask)
            loglikelihood_b_t = diag_gaussian_log_likelihood(dataset_ph,
                                                                  masked_output_mean, masked_output_logvar)

        print('Test3', loglikelihood_b_t.get_shape())
        # cost for each trial

        # log_p_b_train =  (1. / cv_keep_ratio)**2 * tf.reduce_mean(
        #    tf.reduce_mean(loglikelihood_b_t * cv_binary_mask, [2] ), [1] )
        # total rec cost (avg over all trials in batch)
        # log_p_train = tf.reduce_mean( log_p_b_train, [0])
        # rec_cost_train = -log_p_train
        # ones_ratio = tf.reduce_mean(cv_binary_mask)

        rec_cost_heldin_samps = - (1. / cv_keep_ratio) * \
                              tf.reduce_sum(loglikelihood_b_t * cv_binary_mask)/ tf.cast(batch_size, tf.float32)
        # rec_cost_heldin_samps = - tf.reduce_sum(loglikelihood_b_t * cv_binary_mask) / batch_size
        # rec_cost_train /= ones_ratio

        # Valid cost for each trial
        # log_p_b_valid = (1. / (1. - cv_keep_ratio))**2 * tf.reduce_mean(
        #    tf.reduce_mean(loglikelihood_b_t * (1. - cv_binary_mask), [2] ), [1] )
        # total rec cost (avg over all trials in batch)
        # log_p_valid = tf.reduce_mean( log_p_b_valid, [0])
        # rec_cost_heldout_samps = -log_p_valid
        if cv_keep_ratio < 1:
            rec_cost_heldout_samps = - (1. / (1. - cv_keep_ratio)) * \
                                  tf.reduce_sum(loglikelihood_b_t * (1. - cv_binary_mask)) / tf.cast(batch_size, tf.float32)
        else:
            rec_cost_heldout_samps = tf.constant(np.nan)
        # rec_cost_heldout_samps = - tf.reduce_sum(loglikelihood_b_t * (1. - cv_binary_mask)) / batch_size
        # rec_cost_heldout_samps /= (1. - ones_ratio)

        # calculate L2 costs for each network
        # normalized by number of parameters.
        l2_cost = tf.constant(0.0)
        l2_costs = []
        l2_numels = []
        l2_reg_var_lists = ['l2_gen',
                            'l2_con',
                            'l2_ic_enc',
                            'l2_ci_enc',
                            'l2_gen_2_factors',
                            'l2_ci_enc_2_co_in',
                            ]

        l2_reg_scales = [hps['l2_gen_scale'], hps['l2_con_scale'],
                         hps['l2_ic_enc_scale'], hps['l2_ci_enc_scale'],
                         hps['l2_gen_2_factors_scale'],
                         hps['l2_ci_enc_2_co_in']]
        for l2_reg, l2_scale in zip(l2_reg_var_lists, l2_reg_scales):
            if l2_scale == 0:
                continue
            l2_reg_vars = tf.get_collection(l2_reg)
            for v in l2_reg_vars:
                numel = tf.reduce_prod(tf.concat(axis=0, values=tf.shape(v)))
                numel_f = tf.cast(numel, tf.float32)
                l2_numels.append(numel_f)
                v_l2 = tf.reduce_sum(v * v)
                l2_costs.append(0.5 * l2_scale * v_l2)

        if l2_numels:
            l2_cost = tf.add_n(l2_costs) / tf.add_n(l2_numels)

        ## calculate total training cost
        total_cost = l2_cost + kl_cost + rec_cost_heldin_samps

        # get the list of trainable variables
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if hps['do_train_readin'] == False:
            # filter out any variables with name containing '_in_fac_linear'
            regex = re.compile('.+_in_fac_linear.+')
            trainable_vars = [i for i in trainable_vars if not regex.search(i.name)]

        gradients = tf.gradients(total_cost, trainable_vars)
        gradients, grad_global_norm = tf.clip_by_global_norm(gradients, hps['max_grad_norm'])
        # this is the optimizer
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-01)

        if self.use_tpu:
            opt = tf.contrib.tpu.CrossShardOptimizer(opt)

        global_step = tf.train.get_global_step()
        #train_op = opt.minimize(total_cost, global_step=global_step)
        train_op = opt.apply_gradients(
            zip(gradients, trainable_vars), global_step=global_step)

        # print Costs
        # total_cost = tf.self.printlog(total_cost, [tf.train.get_global_step(), total_cost, rec_cost_train])

        # Set logging hook for tf.estimator
        logging_hook = tf.train.LoggingTensorHook({'step': global_step,
                                                   'train_total': total_cost,
                                                   'train_recon_heldin_samps': rec_cost_heldin_samps,
                                                   'train_recon_heldout_samps': rec_cost_heldout_samps,
                                                   'l2_cost': l2_cost,
                                                   'kl_cost': kl_cost,
                                                   },
                                                  every_n_iter=1)
        # metric for eval
        #eval_metric_ops = {
        #    'eval_data_mode': tf.metrics.mean(data_mode),
        #    'eval_recon_heldin_samps': tf.metrics.mean(rec_cost_heldin_samps),
        #    'eval_recon_heldout_samps': tf.metrics.mean(rec_cost_heldout_samps),
        #    'eval_total': tf.metrics.mean(total_cost),
        #}
        print(loglikelihood_b_t.get_shape())        
        metric_rec_cost_heldin_samps = - (1. / cv_keep_ratio) * \
                                       tf.reduce_sum(tf.reduce_sum(loglikelihood_b_t * cv_binary_mask, axis=1),axis=1)

        if cv_keep_ratio != 1.0:
            metric_rec_cost_heldout_samps = - (1. / (1. - cv_keep_ratio)) * \
                                            tf.reduce_sum(tf.reduce_sum(loglikelihood_b_t * (1. - cv_binary_mask), axis=1), axis=1)
        else:
            metric_rec_cost_heldout_samps = tf.constant(np.nan, shape=[batch_size, 1])


        tpu_eval_metrics = None
        if mode == tf.estimator.ModeKeys.EVAL:
            train_op = None
            def metric_fn(metric_rec_cost_heldin_samps, metric_rec_cost_heldout_samps):
                return {
                #'eval_data_mode': tf.metrics.mean(data_mode),
                'eval_recon_heldin_samps': tf.metrics.mean(metric_rec_cost_heldin_samps),
                'eval_recon_heldout_samps': tf.metrics.mean(metric_rec_cost_heldout_samps),
                #'eval_total': tf.metrics.mean(total_cost),
                }
            tpu_eval_metrics = (metric_fn, [metric_rec_cost_heldin_samps, metric_rec_cost_heldout_samps])
            
        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        # for TPU
        return tpu_estimator.TPUEstimatorSpec(mode=mode,
                                          loss=total_cost,
                                          train_op=train_op,
                                          eval_metrics=tpu_eval_metrics,
                                          #eval_metric_ops=eval_metric_ops,
                                          #training_hooks=[logging_hook]
                                          )

        #return tpu_estimator.TPUEstimatorSpec(mode=mode,
        #                                  loss=total_cost,
        #                                  eval_metrics=tpu_eval_metrics,
                                          #eval_metric_ops=eval_metric_ops,
                                          #training_hooks=[logging_hook]
        #                                  )

        #return tf.estimator.EstimatorSpec(mode=mode,
        #                                  loss=total_cost,
        #                                  train_op=train_op,
        #                                  eval_metric_ops=eval_metric_ops,
        #                                  training_hooks=[logging_hook])

    def setup_stdout_logger(self, hps):
        # save the stdout to a log file and prints it on the screen
        mkdir_p(hps['lfads_save_dir'])
        logger = Logger(os.path.join(hps['lfads_save_dir'], "lfads_output.log"))
        self.printlog = logger.printlog

    def setup_tf_logger(self, hps):
        # get TF logger
        log = logging.getLogger('tensorflow')
        log.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = CustomFormatter()
        # create file handler to save logging
        csv_file = os.path.join(hps['lfads_save_dir'], hps['csv_log'] + '.csv')
        fh = logging.FileHandler(csv_file)
        fh.addFilter(CustomFilter())
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log.addHandler(fh)

    def train_model(self, hps, run_mode='standard', num_steps=None):
        params = {}
        for key, value in hps.iteritems():
            params[key] = value
        tf.logging.set_verbosity(tf.logging.INFO)
        #self.setup_stdout_logger(hps)
        #self.setup_tf_logger(hps)
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True)
        hps['n_epochs_eval'] = 10
        # maximum 100 iterations per loop (can change to num_steps later but need testing)
        iterations_per_loop = min(num_steps, 100)
        run_config = tf.contrib.tpu.RunConfig(
            save_checkpoints_steps=500,
            cluster=self.tpu_cluster_resolver,
            #master=self.master,
            #evaluation_master=self.master,
            keep_checkpoint_max=hps['max_ckpt_to_keep'],
            #tf_random_seed=19830610,
            # TODO, MRK, need to fix the model_dir and the log dir
            #model_dir = 'gs://pbt-test-bucket/runs',#'file:/' + hps['lfads_save_dir'],
            model_dir=hps['lfads_save_dir'],
            session_config=config,
            tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1, num_shards=1),
        )
        eval_batch_size = params['eval_batch_size']

        if run_mode.lower() == 'pbt':
            assert num_steps, 'You must specify max_epochs when run_mode is PBT!'
            train_input = self.data_fn(tf.estimator.ModeKeys.TRAIN, data_type='train')
            eval_input_train =  self.data_fn(tf.estimator.ModeKeys.EVAL, data_type='train')
            eval_input_valid = self.data_fn(tf.estimator.ModeKeys.EVAL, data_type='valid')
            lfads = tf.contrib.tpu.TPUEstimator(model_fn=self.lfads_model_fn, params=params, config=run_config,
                use_tpu=self.use_tpu, train_batch_size=params['train_batch_size'], eval_batch_size=eval_batch_size,
                predict_batch_size=params['train_batch_size'],)
            #lr = None #self.learning_rate if hps['do_reset_learning_rate'] else None
            #train_hook = TrainHook(lr)
            #, hooks = [train_hook]
            lfads.train(train_input, steps=num_steps)
            # TODO, MRK, calcualtes the steps for evaluation through the data once
            train_costs = lfads.evaluate(eval_input_train, name='train_data', steps=2)
            valid_costs = lfads.evaluate(eval_input_valid, name='valid_data', steps=2)

            #self.printlog('Training costs:', train_costs)
            #self.printlog('Validation costs:', valid_costs)
            print('Training costs:', train_costs)
            print('Validation costs:', valid_costs)

            # Making parameters available for lfads_wrappper
            if hps['checkpoint_pb_load_name'] == 'checkpoint_lve':
                # if we are returning lve costs
                assert 0, 'lve is not implemented with TPU tf.estimator implementation'
                #if is_lve:
                #    self.trial_recon_cost = smth_trial_val_recn_cost
                #    self.samp_recon_cost = smth_samp_val_recn_cost
            else:
                self.trial_recon_cost = valid_costs['eval_recon_heldin_samps']
                self.samp_recon_cost = train_costs['eval_recon_heldout_samps']
        else:
            # MRK, this part has not been tested nor developed for the time being
            train_input = self.data_fn(tf.estimator.ModeKeys.TRAIN, data_type='train')
            eval_input_train =  self.data_fn(tf.estimator.ModeKeys.EVAL, data_type='train')
            eval_input_valid = self.data_fn(tf.estimator.ModeKeys.EVAL, data_type='valid')
            lfads = tf.contrib.tpu.TPUEstimator(model_fn=self.lfads_model_fn, config=run_config,
                use_tpu=self.use_tpu, train_batch_size=params['train_batch_size'], eval_batch_size=eval_batch_size)
            #lfads = tf.estimator.Estimator(model_fn=self.lfads_model_fn, params=hps, config=run_config)
            train_spec = tf.estimator.TrainSpec(train_input, max_steps=num_steps)
            eval_spec = tf.estimator.EvalSpec(eval_input_train,
                                              #start_delay_secs=10,
                                              steps=1,
                                              #throttle_secs = 50
                                              )
                                              #exporters=[exporter],
                                              #name='census-eval')

            tf.estimator.train_and_evaluate(lfads, train_spec, eval_spec)
            #tf.estimator
        #lfads.train(input_fn=data_fn)
        #eval_out = lfads.evaluate(input_fn=data_fn, steps=1)
        #self.printlog(eval_out)
        #pred_out = lfads.predict(input_fn=data_fn)
        #self.printlog(pred_out['output_dist_params'])



    def eval_model_runs_batch(self, data_name, data_bxtxd,
                              do_eval_cost=False, do_average_batch=False):
        """Returns all the goodies for the entire model, per batch.

        Args:
          data_name: The name of the data dict, to select which in/out matrices
            to use.
          data_bxtxd:  Numpy array training data with shape:
            batch_size x # time steps x # dimensions
          ext_input_bxtxi: Numpy array training external input with shape:
            batch_size x # time steps x # external input dims
          do_eval_cost (optional): If true, the IWAE (Importance Weighted
             Autoencoder) log likeihood bound, instead of the VAE version.
          do_average_batch (optional): average over the batch, useful for getting
          good IWAE costs, and model outputs for a single data point.

        Returns:
          A dictionary with the outputs of the model decoder, namely:
            prior g0 mean, prior g0 variance, approx. posterior mean, approx
            posterior mean, the generator initial conditions, the control inputs (if
            enabled), the state of the generator, the factors, and the rates.
        """
        session = tf.get_default_session()
        # kl_ic_weight and kl_co_weight do not matter for posterior sample and average
        if do_average_batch:
            run_type = kind_dict('posterior_mean')
        else:
            run_type = kind_dict('posterior_sample_and_average')

        feed_dict = self.build_feed_dict(data_name, data_bxtxd, run_type=run_type,
                                         keep_prob=1.0, keep_ratio=1.0)
        # Non-temporal signals will be batch x dim.
        # Temporal signals are list length T with elements batch x dim.
        tf_vals = [self.gen_ics, self.gen_states, self.factors,
                   self.output_dist_params]
        if self.hps.ic_dim > 0:
          tf_vals += [self.prior_zs_g0.mean, self.prior_zs_g0.logvar,
                      self.posterior_zs_g0.mean, self.posterior_zs_g0.logvar]
        if self.hps.co_dim > 0:
          tf_vals.append(self.controller_outputs)

        # flatten for sending into session.run
        np_vals_flat = session.run(tf_vals, feed_dict=feed_dict)
        return np_vals_flat
    #        tf_vals_flat, fidxs = flatten(tf_vals)


    # this does the bulk of posterior sample & mean
    def eval_model_runs_avg_epoch(self, data_name, data_extxd, pm_batch_size=None,
                                  do_average_batch=False):
        """Returns all the expected value for goodies for the entire model.

        The expected value is taken over hidden (z) variables, namely the initial
        conditions and the control inputs.  The expected value is approximate, and
        accomplished via sampling (batch_size) samples for every examples.

        Args:
          data_name: The name of the data dict, to select which in/out matrices
            to use.
          data_extxd:  Numpy array training data with shape:
            # examples x # time steps x # dimensions
#          ext_input_extxi (optional): Numpy array training external input with
#            shape: # examples x # time steps x # external input dims

        Returns:
          A dictionary with the averaged outputs of the model decoder, namely:
            prior g0 mean, prior g0 variance, approx. posterior mean, approx
            posterior mean, the generator initial conditions, the control inputs (if
            enabled), the state of the generator, the factors, and the output
            distribution parameters, e.g. (rates or mean and variances).
        """
        hps = self.hps
        batch_size = hps.batch_size
        if pm_batch_size is None:
            pm_batch_size = batch_size
        E, T, D  = data_extxd.shape
        E_to_process = hps.ps_nexamples_to_process
        if E_to_process > E:
          #self.printlog("Setting number of posterior samples to process to : %d" % E)
          print("Setting number of posterior samples to process to : %d" % E)
          E_to_process = E

        # make a bunch of placeholders to store the posterior sample means
        gen_ics = np.array([]) # np.zeros([E_to_process, hps.gen_dim])
        gen_states = np.array([]) #np.zeros([E_to_process, T, hps.gen_dim])
        factors = np.array([]) #np.zeros([E_to_process, T, hps.factors_dim])
        out_dist_params = np.array([]) #np.zeros([E_to_process, T, D])
        if hps.ic_dim > 0:
            prior_g0_mean = np.array([]) #np.zeros([E_to_process, hps.ic_dim])
            prior_g0_logvar = np.array([]) #np.zeros([E_to_process, hps.ic_dim])
            post_g0_mean = np.array([]) #np.zeros([E_to_process, hps.ic_dim])
            post_g0_logvar = np.array([]) #np.zeros([E_to_process, hps.ic_dim])

        if hps.co_dim > 0:
            controller_outputs = np.array([]) #np.zeros([E_to_process, T, hps.co_dim])

        #costs = np.zeros(E_to_process)
        #nll_bound_vaes = np.zeros(E_to_process)
        #nll_bound_iwaes = np.zeros(E_to_process)
        #train_steps = np.zeros(E_to_process)
        # MRK change the over of processing posterior samples
        # batches are trials

        # choose E_to_process trials from the data
        data_bxtxd = data_extxd[0:E_to_process]
        for rep in range(pm_batch_size):
            printer("Running repetitions %d of %d." % (rep + 1, pm_batch_size))
            #self.printlog("Running %d of %d." % (es_idx+1, E_to_process))
            #example_idxs = es_idx * np.ones(batch_size, dtype=np.int32)
            # run the model
            mv = self.eval_model_runs_batch(data_name, data_bxtxd,
                                            do_eval_cost=True,
                                            do_average_batch=do_average_batch)
            # assemble a dict from the returns
            model_values = {}
            # compile a list of variables "vars"
            vars = ['gen_ics', 'gen_states', 'factors', 'output_dist_params']
            if self.hps.ic_dim > 0:
                vars.append('prior_g0_mean')
                vars.append('prior_g0_logvar')
                vars.append('post_g0_mean')
                vars.append('post_g0_logvar')
            if self.hps.co_dim > 0:
                vars.append('controller_outputs')

            # put returned variables that were on the "vars" list into a "model_values" dict
            for idx in range( len(vars) ):
                model_values[ vars[idx] ] = mv[idx] / pm_batch_size

            # CP: the below used to append, and do a mean later
            #   now switching to do the averaging in the loop to save memory

            # assign values to the arrays
            #  if it's the first go 'round:
            if not gen_ics.size:
                gen_ics = model_values['gen_ics']
                gen_states = model_values['gen_states']
                factors = model_values['factors']
                out_dist_params = model_values['output_dist_params']
                if self.hps.ic_dim > 0:
                    prior_g0_mean = model_values['prior_g0_mean']
                    prior_g0_logvar = model_values['prior_g0_logvar']
                    post_g0_mean = model_values['post_g0_mean']
                    post_g0_logvar = model_values['post_g0_logvar']
                if self.hps.co_dim > 0:
                    controller_outputs = model_values['controller_outputs']
            else:
                gen_ics = gen_ics + model_values['gen_ics']
                gen_states = gen_states + model_values['gen_states']
                factors = factors + model_values['factors']
                out_dist_params = out_dist_params + model_values['output_dist_params']
                if self.hps.ic_dim > 0:
                    prior_g0_mean = prior_g0_mean + model_values['prior_g0_mean']
                    prior_g0_logvar = prior_g0_logvar + model_values['prior_g0_logvar']
                    post_g0_mean = post_g0_mean + model_values['post_g0_mean']
                    post_g0_logvar = post_g0_logvar + model_values['post_g0_logvar']
                if self.hps.co_dim > 0:
                    controller_outputs = controller_outputs + model_values['controller_outputs']

        #self.printlog("")
        print("")
        model_runs = {}
        model_runs['gen_ics'] = gen_ics
        model_runs['gen_states'] = gen_states
        model_runs['factors'] = factors
        model_runs['output_dist_params'] = out_dist_params
        if self.hps.ic_dim > 0:
            model_runs['prior_g0_mean'] = prior_g0_mean
            model_runs['prior_g0_logvar'] = prior_g0_logvar
            model_runs['post_g0_mean'] = post_g0_mean
            model_runs['post_g0_logvar'] = post_g0_logvar

        if self.hps.co_dim > 0:
            model_runs['controller_outputs'] = controller_outputs
                                                   
        # return the dict
        return model_runs

    '''
    def get_R2(self, data_true, data_est, mask):
        """Calculate the R^2 between the true rates and LFADS output, over all
        trials and all channels
        """
        true_flat = data_true.flatten()
        est_flat = data_est.flatten()

        #plt.plot(true_flat)
        #plt.plot(est_flat)
        #plt.show()

        mask = np.repeat(np.expand_dims(mask, axis=0), data_true.shape[0], axis=0)
        mask = mask.flatten()
        mask = mask.astype(np.bool)

        R2_heldin = np.corrcoef(true_flat[mask], est_flat[mask])**2.0
        R2_heldout = np.corrcoef(true_flat[np.invert(mask)], est_flat[np.invert(mask)])**2.0

        return R2_heldin[0,1], R2_heldout[0,1]
    '''

    # this calls self.eval_model_runs_avg_epoch to get the posterior means
    # then it writes all the data to file
    def write_model_runs(self, datasets, output_fname=None):
        """Run the model on the data in data_dict, and save the computed values.

        LFADS generates a number of outputs for each examples, and these are all
        saved.  They are:
          The mean and variance of the prior of g0.
          The mean and variance of approximate posterior of g0.
          The control inputs (if enabled)
          The initial conditions, g0, for all examples.
          The generator states for all time.
          The factors for all time.
          The output distribution parameters (e.g. rates) for all time.

        Args:
          datasets: a dictionary of named data_dictionaries, see top of lfads.py
          output_fname: a file name stem for the output files.
        """
        hps = self.hps
        kind = kind_dict_key(hps.kind)
        all_model_runs = []

        for data_name, data_dict in datasets.items():
          data_tuple = [('train', data_dict['train_data']),
                        ('valid', data_dict['valid_data'])]
          for data_kind, data_extxd in data_tuple:
            if not output_fname:
              fname = "model_runs_" + data_name + '_' + data_kind + '_' + kind
            else:
              fname = output_fname + data_name + '_' + data_kind + '_' + kind

            #self.printlog("Writing data for %s data and kind %s to file %s." % (data_name, data_kind, fname))
            print("Writing data for %s data and kind %s to file %s." % (data_name, data_kind, fname))
            model_runs = self.eval_model_runs_avg_epoch(data_name, data_extxd)
            all_model_runs.append(model_runs)
            full_fname = os.path.join(hps.lfads_save_dir, fname)
            write_data(full_fname, model_runs, compression='gzip')
            #self.printlog("Done.")
            print("Done.")
        return all_model_runs

            
