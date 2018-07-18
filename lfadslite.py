from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import time
import os
import re
import random
#import matplotlib.pyplot as plt


# utils defined by CP
from helper_funcs import linear, init_linear_transform, makeInitialState
from helper_funcs import ListOfRandomBatches, kind_dict, kind_dict_key
from helper_funcs import DiagonalGaussianFromInput, DiagonalGaussian, LearnableAutoRegressive1Prior
from helper_funcs import DiagonalGaussianFromExisting, LearnableDiagonalGaussian, diag_gaussian_log_likelihood
from helper_funcs import LinearTimeVarying
from helper_funcs import KLCost_GaussianGaussian, KLCost_GaussianGaussianProcessSampled
from helper_funcs import write_data
from helper_funcs import printer, mkdir_p
#from plot_funcs import plot_data, close_all_plots
#from data_funcs import read_datasets
from customcells import ComplexCell
from rnn_helper_funcs import BidirectionalDynamicRNN, DynamicRNN
from helper_funcs import dropout


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

#class Logger(object):
#    def __init__(self, log_file):
#        self.terminal = sys.stdout
#        self.log = open(log_file, "a")#
#
#    def write(self, message):
#        self.terminal.write(message)
#        self.log.write(message)  
#
#    def flush(self):
#        #this flush method is needed for python 3 compatibility.
#        #this handles the flush command by doing nothing.
#        #you might want to specify some extra behavior here.
#        pass    

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

class LFADS(object):

    def __init__(self, hps, datasets = None):

        #CELL_TYPE = 'lstm'
        #CELL_TYPE = 'gru'
        CELL_TYPE = 'customgru'

        # to stop certain gradients paths through the graph in backprop
        def entry_stop_gradients(target, mask):
            mask_h = 1. - mask
            return tf.stop_gradient(mask_h * target) + mask * target

        # save the stdout to a log file and prints it on the screen
        mkdir_p(hps['lfads_save_dir'])
        #sys.stdout = Logger(os.path.join(hps['lfads_save_dir'], "lfads_output.log"))
        logger = Logger(os.path.join(hps['lfads_save_dir'], "lfads_output.log"))
        self.printlog = logger.printlog

        # build the graph
        # set the learning rate, defaults to the hyperparam setting
        self.learning_rate = tf.Variable(float(hps['learning_rate_init']), trainable=False, name="learning_rate")

        # this is how the learning rate is decayed over time
        self.learning_rate_decay_op = self.learning_rate.assign(\
            self.learning_rate * hps['learning_rate_decay_factor'])

        ### BEGIN MODEL CONSTRUCTION

        # NOTE: the graph differs slightly on the input side depending on whether there are multiple datasets or not
        #  if multiple datasets (or if input_factors_dim is defined), there must be an 'input factors' layer
        #  - this sets a common dimension across datasets, allowing datasets to have different sizes
        #  if not multiple datasets and no input_factors_dim is defined, we'll hook data straight to encoders

        # define all placeholders
        with tf.variable_scope('placeholders'):
            # input data (what are we training on)
            # we're going to try setting input dimensionality to None
            #  so datasets with different sizes can be used
            self.dataset_ph = tf.placeholder(tf.float32, shape = [None, hps['num_steps'], None], name='input_data')
            self.cv_rand_mask = tf.placeholder(tf.float32, shape=[None, hps['num_steps'], None], name='cv_rand_mask')
            # dropout keep probability
            #   enumerated in helper_funcs.kind_dict
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.keep_ratio = tf.placeholder(tf.float32, name='keep_ratio')

            self.run_type = tf.placeholder(tf.int32, name='run_type')
            self.kl_ic_weight = tf.placeholder(tf.float32, name='kl_ic_weight')
            self.kl_co_weight = tf.placeholder(tf.float32, name='kl_co_weight')
            # ramp KL and L2 cost weights
            self.kl_weight = tf.placeholder(tf.float32, name='kl_weight')
            self.l2_weight = tf.placeholder(tf.float32, name='l2_weight')

            # name of the dataset
            self.dataName = tf.placeholder(tf.string, shape=(), name='dataset_name')
            if hps['ext_input_dim'] > 0:
                self.ext_input = tf.placeholder(tf.float32,
                                      [None, hps['num_steps'], hps['ext_input_dim']],
                                      name="ext_input")
            else:
                self.ext_input = None

        if hps['ext_input_dim'] > 0:
            self.ext_input = tf.nn.dropout(self.ext_input, self.keep_prob)
        else:
            self.ext_input = None


        # make placeholders for all the input and output adapter matrices
        ndatasets = hps.ndatasets
        # preds will be used to select elements of each session
        self.preds = preds = [None] * ndatasets
        self.fns_in_fac_Ws = fns_in_fac_Ws = [None] * ndatasets
        self.fns_in_fac_bs = fns_in_fac_bs = [None] * ndatasets
        self.fns_out_fac_Ws = fns_out_fac_Ws = [None] * ndatasets
        self.fns_out_fac_bs = fns_out_fac_bs = [None] * ndatasets
        self.datasetNames = dataset_names = hps.dataset_names
        # specific to lfadslite - need to make placeholders for the cross validation dropout masks
        #dataset_dims = [None] * ndatasets
        fns_this_dataset_dims = [None] * ndatasets

        
        # figure out the input (dataset) dimensionality
        #allsets = hps['dataset_dims'].keys()
        #self.input_dim = hps['dataset_dims'][allsets[0]]
        
        self.cv_keep_ratio = hps['cv_keep_ratio']
        self.cd_grad_passthru_prob = hps['cd_grad_passthru_prob']

        ## do per-session stuff
        for d, name in enumerate( dataset_names ):
            data_dim = hps.dataset_dims[name]

            # Step 0) define the preds comparator for this dataset
            preds[ d ] = tf.equal( tf.constant( name ), self.dataName )
            
            # Step 1) alignment matrix stuff.
            # the alignment matrix only matters if in_factors_dim is nonzero
            in_mat_cxf = None
            align_bias_1xc = None
            in_bias_1xf = None
            if hps.in_factors_dim > 0:
                # get the alignment_matrix if provided
                if 'alignment_matrix_cxf' in datasets[ name ].keys():
                    in_mat_cxf = datasets[ name ][ 'alignment_matrix_cxf'].astype( np.float32 )
                    # check that sizing is appropriate
                    if in_mat_cxf.shape != (data_dim, hps.in_factors_dim):
                        raise ValueError("""Alignment matrix must have dimensions %d x %d
                        (data_dim x factors_dim), but currently has %d x %d."""%
                                         (data_dim, hps.in_factors_dim, in_mat_cxf.shape[0],
                                          in_mat_cxf.shape[1]))
                if 'alignment_bias_c' in datasets[ name ].keys():
                    align_bias_c = datasets[ name ][ 'alignment_bias_c'].astype( np.float32 )
                    align_bias_1xc = np.expand_dims(align_bias_c, axis=0)
                    if align_bias_1xc.shape[1] != data_dim:
                        raise ValueError("""Alignment bias must have dimensions %d
                        (data_dim), but currently has %d."""%
                                         (data_dim, in_mat_cxf.shape[0]))
                if in_mat_cxf is not None and align_bias_1xc is not None:
                    # (data - alignment_bias) * W_in
                    # data * W_in - alignment_bias * W_in
                    # So b = -alignment_bias * W_in to accommodate PCA style offset.
                    in_bias_1xf = -np.dot(align_bias_1xc, in_mat_cxf)
                # initialize a linear transform based on the above
                in_fac_linear = init_linear_transform( data_dim, hps.in_factors_dim, mat_init_value=in_mat_cxf,
                                                       bias_init_value=in_bias_1xf,
                                                       name= name+'_in_fac_linear' )
                in_fac_W, in_fac_b = in_fac_linear
                # to store per-session matrices/biases for later use, need to use 'makelambda'
                fns_in_fac_Ws[d] = makelambda(in_fac_W)
                fns_in_fac_bs[d] = makelambda(in_fac_b)
                
            # single-sample cross-validation mask
            # generate one random mask once (for each dataset) when building the graph
            # use a different (but deterministic) random seed for each dataset (hence adding 'd' below)
            #if hps.cv_rand_seed:
            #    np.random.seed( int(hps.cv_rand_seed) + d)

            # Step 2) make a get the dataset dim (work around dim error in dynamic rnn)
            #dataset_dims[ d ] = hps.dataset_dims[ name ]
            # converting to tensor
            fns_this_dataset_dims[ d ] = makelambda( tf.ones((hps['num_steps'], hps.dataset_dims[ name ])) )

            #reset the np random seed to enforce randomness for the other random draws
            #np.random.seed()
            
            # Step 3) output matrix stuff
            out_mat_fxc = None
            out_bias_1xc = None
            
            # if input and output factors dims match, can initialize output matrices using transpose of input matrices
            if in_mat_cxf is not None:
                if hps.in_factors_dim==hps.factors_dim:
                    out_mat_fxc = in_mat_cxf.T
            if align_bias_1xc is not None:
                out_bias_1xc = align_bias_1xc
            
            if hps.output_dist.lower() == 'poisson':
                output_size = data_dim
            elif hps.output_dist.lower() == 'gaussian':
                output_size = data_dim * 2
                if out_mat_fxc is not None:
                    out_mat_fxc = tf.concat( [ out_mat_fxc, out_mat_fxc ], 0 )
                if out_bias_1xc is not None:
                    out_bias_1xc = tf.concat( [ out_bias_1xc, out_bias_1xc ], 0 )
            out_fac_linear = init_linear_transform( hps.factors_dim, output_size, mat_init_value=out_mat_fxc,
                                                   bias_init_value=out_bias_1xc,
                                                   name= name+'_out_fac_linear' )
            out_fac_W, out_fac_b = out_fac_linear
            fns_out_fac_Ws[d] = makelambda(out_fac_W)
            fns_out_fac_bs[d] = makelambda(out_fac_b)

        # now 'zip' together the 'pred' selector with all the function handles
        pf_pairs_in_fac_Ws = zip(preds, fns_in_fac_Ws)
        pf_pairs_in_fac_bs = zip(preds, fns_in_fac_bs)
        pf_pairs_out_fac_Ws = zip(preds, fns_out_fac_Ws)
        pf_pairs_out_fac_bs = zip(preds, fns_out_fac_bs)
        pf_pairs_this_dataset_dims = zip(preds, fns_this_dataset_dims )

        # now, choose the ones for this session
        if hps.in_factors_dim > 0:
            this_dataset_in_fac_W = _case_with_no_default( pf_pairs_in_fac_Ws )
            this_dataset_in_fac_b = _case_with_no_default( pf_pairs_in_fac_bs )
            
        this_dataset_out_fac_W = _case_with_no_default( pf_pairs_out_fac_Ws )
        this_dataset_out_fac_b = _case_with_no_default( pf_pairs_out_fac_bs )
        this_dataset_dims = _case_with_no_default( pf_pairs_this_dataset_dims )
                
        # apply dropout to the data
        self.dataset_ph = tf.nn.dropout(self.dataset_ph, self.keep_prob)
        # batch_size - read from the data placeholder
        graph_batch_size = tf.shape(self.dataset_ph)[0]
        # can we infer the data dimensionality for the random mask?
        seq_len = hps['num_steps']

        # coordinated dropout
        if hps['keep_ratio'] != 1.0:
            # coordinated dropout enabled on inputs
            masked_dataset_ph, coor_drop_binary_mask = dropout(self.dataset_ph, self.keep_ratio)
        else:
            # no coordinated dropout
            masked_dataset_ph = self.dataset_ph

        # replicate the cross-validation binary mask for this dataset for all elements of the batch
        # work around error in dynamic rnn when input dim is None

        self.cv_binary_mask_batch = self.cv_rand_mask * tf.expand_dims(tf.ones([graph_batch_size, 1]), 1) * \
                                    this_dataset_dims

        # apply cross-validation dropout
        masked_dataset_ph = tf.div(masked_dataset_ph, self.cv_keep_ratio) * self.cv_binary_mask_batch

        # define input to encoders
        if hps.in_factors_dim==0:
            self.input_to_encoders = masked_dataset_ph
        else:
            input_factors_object = LinearTimeVarying(inputs = masked_dataset_ph,
                                                    output_size = hps.in_factors_dim,
                                                    transform_name = 'data_2_infactors',
                                                    output_name = 'infactors',
                                                    W = this_dataset_in_fac_W,
                                                    b = this_dataset_in_fac_b,
                                                    nonlinearity = None)
            self.input_to_encoders = input_factors_object.output
            
        with tf.variable_scope('ic_enc'):
            #ic_enc_cell = CustomGRUCell(num_units = hps['ic_enc_dim'],\
            #                            batch_size = graph_batch_size,
            #                            clip_value = hps['cell_clip_value'],
            #                            recurrent_collections=['l2_ic_enc'])
            
            #seq_len.set_shape([1, graph_batch_size])
            ## ic_encoder
            self.ic_enc_rnn_obj = BidirectionalDynamicRNN(
                state_dim = hps['ic_enc_dim'],
                batch_size = graph_batch_size,
                name = 'ic_enc',
                sequence_lengths = seq_len,
                inputs = self.input_to_encoders,
                initial_state = None,
                clip_value = hps['cell_clip_value'],
                recurrent_collections='l2_ic_enc',
                rnn_type = CELL_TYPE)
            #    cell = ic_enc_cell
            #    output_keep_prob = self.keep_prob

            # wrap the last state with a dropout layer
            #ic_enc_laststate_dropped = self.ic_enc_rnn_obj.last_tot
            ic_enc_laststate_dropped = tf.nn.dropout(self.ic_enc_rnn_obj.last_tot, self.keep_prob)
            
            # map the ic_encoder onto the actual ic layer
            self.gen_ics_posterior = DiagonalGaussianFromInput(
                x = ic_enc_laststate_dropped,
                z_size = hps['ic_dim'],
                name = 'ic_enc_2_ics',
                var_min = hps['ic_post_var_min'],
            )
            self.posterior_zs_g0 = self.gen_ics_posterior

        # to go forward, either sample from the posterior, or take mean
        do_posterior_sample = tf.logical_or(tf.equal(self.run_type, tf.constant(kind_dict("train"))),
            tf.equal(self.run_type, tf.constant(kind_dict("posterior_sample_and_average"))))
        self.gen_ics_lowd = tf.cond(do_posterior_sample, lambda:self.gen_ics_posterior.sample(), 
            lambda:self.gen_ics_posterior.mean)


        
        with tf.variable_scope('generator'):
            # lstms have twice the number of state dims as everybody else (h and c cells) - correct for that here.
            if CELL_TYPE.lower() == 'lstm':
                self.gen_ics = linear(self.gen_ics_lowd, hps['gen_dim']*2, name='ics_2_g0')
            else:
                self.gen_ics = linear(self.gen_ics_lowd, hps['gen_dim'], name='ics_2_g0')


        ### CONTROLLER construction
        # this should only be done if a controller is requested
        # if not, skip all these graph elements like so:
        if hps['co_dim'] == 0:
            with tf.variable_scope('generator'):
                #gen_cell = CustomGRUCell(num_units = hps['gen_dim'],
                #                         batch_size = graph_batch_size,
                #                         clip_value = hps['cell_clip_value'],
                #                         recurrent_collections=['l2_gen'])
                
                # setup generator
                # will be None with no inputs
                gen_input = self.ext_input

                self.gen_rnn_obj = DynamicRNN(state_dim = hps['gen_dim'],
                                              batch_size = graph_batch_size,
                                              name = 'gen',
                                              sequence_lengths = seq_len,
                                              inputs = gen_input,
                                              initial_state = self.gen_ics,
                                              rnn_type = CELL_TYPE,
                                              recurrent_collections='l2_gen',
                                              clip_value = hps['cell_clip_value']
                )
                #    output_keep_prob = self.keep_prob
                #                              cell = gen_cell,
                
                self.gen_states = self.gen_rnn_obj.states

            with tf.variable_scope('factors'):
                # wrap the generator states in a dropout layer
                #gen_states_dropped = self.gen_rnn_obj.states
                gen_states_dropped = tf.nn.dropout(self.gen_rnn_obj.states, self.keep_prob)
                ## factors
                self.fac_obj = LinearTimeVarying(inputs = gen_states_dropped,
                                                 output_size = hps['factors_dim'],
                                                 transform_name = 'gen_2_factors',
                                                 output_name = 'factors_concat',
                                                 collections='l2_gen_2_factors',
                                                 do_bias = False,
                                                 normalized=True
                )
                self.factors = self.fac_obj.output
        else:
            with tf.variable_scope('ci_enc'):

                #ci_enc_cell = CustomGRUCell(num_units = hps['ci_enc_dim'],\
                #                            batch_size = graph_batch_size,
                #                            clip_value = hps['cell_clip_value'],
                #                            recurrent_collections=['l2_ci_enc'])
                ## ci_encoder
                self.ci_enc_rnn_obj = BidirectionalDynamicRNN(
                    state_dim = hps['ci_enc_dim'],
                    batch_size = graph_batch_size,
                    name = 'ci_enc',
                    sequence_lengths = seq_len,
                    inputs = self.input_to_encoders,
                    initial_state = None,
                    rnn_type = 'gru',
                    recurrent_collections='l2_ci_enc',
                    clip_value = hps['cell_clip_value'])
                
                #    output_keep_prob = self.keep_prob
                #    cell = ci_enc_cell,

                if not hps['do_causal_controller']:
                    self.ci_enc_rnn_states = self.ci_enc_rnn_obj.states
                else:
                    # if causal controller, only use the fwd rnn
                    [ci_enc_fwd_states, _]  = self.ci_enc_rnn_obj.states
                    if hps['controller_input_lag'] > 0:
                        toffset = hps['controller_input_lag']
                        leading_zeros = tf.zeros_like(ci_enc_fwd_states[:,0:toffset,:])
                        self.ci_enc_rnn_states = tf.concat([leading_zeros,
                                                            ci_enc_fwd_states[:,0:-toffset,:]],
                                                           axis=1)
                    else:
                        [self.ci_enc_rnn_states, _]  = self.ci_enc_rnn_obj.states
                
                # states from bidirec RNN come out as a tuple. concatenate those:
                self.ci_enc_rnn_states = tf.concat(self.ci_enc_rnn_states,
                                                   axis=2, name='ci_enc_rnn_states')

                ## take a linear transform of the ci_enc output
                #    this is to lower the dimensionality of the ci_enc
                #with tf.variable_scope('ci_enc_2_co_in'):
                #    # one input from the encoder, another will come back from factors
                #    self.ci_enc_object = LinearTimeVarying(
                #        inputs=self.ci_enc_rnn_states,
                #        output_size = hps['con_ci_enc_in_dim'],
                #        transform_name = 'ci_enc_2_co_in',
                #       output_name = 'ci_enc_output_concat',
                #        nonlinearity = None,
                #        collections='l2_ci_enc_2_co_in')
                #    self.ci_enc_outputs = self.ci_enc_object.output

                # pass ci_enc output directly to the controller
                self.ci_enc_outputs = self.ci_enc_rnn_states
            ## the controller, controller outputs, generator, and factors are implemented
            #     in one RNN whose individual cell is "complex"
            #  this is required do to feedback pathway from factors->controller.
            #    impossible to dynamically unroll with separate RNNs.
            with tf.variable_scope('complexcell'):
                # the "complexcell" architecture requires an initial state definition
                # have to define states for each of the components, then concatenate them
                con_init_state = makeInitialState(hps['con_dim'],
                                                  graph_batch_size,
                                                  'controller')
                co_mean_init_state = makeInitialState(hps['co_dim'],
                                                      graph_batch_size,
                                                      'controller_output_mean')
                co_logvar_init_state = makeInitialState(hps['co_dim'],
                                                        graph_batch_size,
                                                        'controller_output_logvar')
                co_sample_init_state = makeInitialState(hps['co_dim'],
                                                        graph_batch_size,
                                                        'controller_output_sample')
                fac_init_state = makeInitialState(hps['factors_dim'],
                                                  graph_batch_size,
                                                  'factor')
                comcell_init_state = [con_init_state, self.gen_ics,
                                           co_mean_init_state, co_logvar_init_state,
                                           co_sample_init_state, fac_init_state]
                self.complexcell_init_state = tf.concat(axis=1, values = comcell_init_state)


                # here is what the state vector will look like
                self.comcell_state_dims = [hps['gen_dim'],
                                           hps['con_dim'],
                                           hps['co_dim'], # for the controller output means
                                           hps['co_dim'], # for the variances
                                           hps['co_dim'], # for the sampled controller output
                                           hps['factors_dim']]


                # construct the complexcell
                self.complexcell=ComplexCell(num_units_gen=hps['gen_dim'],
                                             num_units_con=hps['con_dim'],
                                             factors_dim=hps['factors_dim'],
                                             co_dim=hps['co_dim'],
                                             ext_input_dim=hps['ext_input_dim'],
                                             inject_ext_input_to_gen=True,
                                             var_min = hps['co_post_var_min'],
                                             run_type = self.run_type,
                                             keep_prob=self.keep_prob)

                # construct the actual RNN
                #   its inputs are the output of the controller_input_enc

                if hps['ext_input_dim']:
                    complex_cell_inputs = tf.concat(axis=2, values = [self.ci_enc_outputs, self.ext_input])
                else:
                    complex_cell_inputs = self.ci_enc_outputs
                self.complex_outputs, self.complex_final_state =\
                tf.nn.dynamic_rnn(self.complexcell,
                                  inputs = complex_cell_inputs,
                                  initial_state = self.complexcell_init_state)

                # split the states of the individual RNNs
                # from the packed "complexcell" state
                self.gen_states, self.con_states, self.co_mean_states, self.co_logvar_states, self.controller_outputs, self.factors =\
                tf.split(self.complex_outputs,
                         self.comcell_state_dims,
                         axis=2)

        # now back to code that runs for all models
        with tf.variable_scope('rates'):
            ## "rates" - more properly called "output_distribution"
            if hps.output_dist.lower() == 'poisson':
                nonlin = 'exp'
            elif hps.output_dist.lower() == 'gaussian':
                nonlin = None
            else:
                raise NameError("Unknown output distribution: " + hps.output_dist)
                
            # rates are taken as a linear (or nonlinear) readout from the factors
            rates_object = LinearTimeVarying(inputs = self.factors,
                                             output_size = output_size,
                                             transform_name = 'factors_2_rates',
                                             output_name = 'rates_concat',
                                             W = this_dataset_out_fac_W,
                                             b = this_dataset_out_fac_b,
                                             nonlinearity = nonlin)

            # select the relevant part of the output depending on model type
            if hps.output_dist.lower() == 'poisson':
                # get both the pre-exponentiated and exponentiated versions
                self.logrates=rates_object.output
                self.output_dist_params=rates_object.output_nl
            elif hps.output_dist.lower() == 'gaussian':
                # get linear outputs, split into mean and variance
                self.output_mean, self.output_logvar = tf.split(rates_object.output,
                                                                2, axis=2)
                self.output_dist_params=rates_object.output


        ## calculate the KL cost
        # g0 - build a prior distribution to compare to
        self.gen_ics_prior = LearnableDiagonalGaussian(
            batch_size=graph_batch_size,
            z_size = [hps['ic_dim']], name='gen_ics_prior',
            var = hps['ic_prior_var'])
        self.prior_zs_g0 = self.gen_ics_prior
        
        # g0 KL cost for each trial
        self.kl_cost_g0_b = KLCost_GaussianGaussian(self.gen_ics_posterior,
                                                    self.gen_ics_prior).kl_cost_b
        # g0 KL cost for the whole batch
        self.kl_cost_g0 = tf.reduce_mean(self.kl_cost_g0_b)
        # total KL cost
        self.kl_cost = self.kl_cost_g0 * self.kl_ic_weight
        if hps['co_dim'] > 0:
            # if there are controller outputs, calculate a KL cost for them
            # first build a prior to compare to
            # Controller outputs
            #autocorrelation_taus = [hps.prior_ar_atau for x in range(hps.co_dim)]
            #noise_variances = [hps.prior_ar_nvar for x in range(hps.co_dim)]
            #self.cos_prior = prior_zs_ar_con = \
            #    LearnableAutoRegressive1Prior(graph_batch_size, hps.co_dim,
            #                                  autocorrelation_taus,
            #                                  noise_variances,
            #                                  hps.do_train_prior_ar_atau,
            #                                  hps.do_train_prior_ar_nvar,
            #                                  hps['num_steps'], "u_prior_ar1")

            # zero mean DiagonalGaussian
            self.cos_prior = DiagonalGaussian(
                z_size = [graph_batch_size, hps['num_steps'], hps['co_dim']],
                name='cos_prior', var = hps['co_prior_var'])

            # then build a posterior
            self.cos_posterior = DiagonalGaussianFromExisting(
                self.co_mean_states,
                self.co_logvar_states)

            #print(self.co_mean_states)
            # MRK, changed this to LFADS'
            #self.kl_cost_co_b_t = \
            #    KLCost_GaussianGaussianProcessSampled(
            #        self.cos_posterior, self.cos_prior).kl_cost_b

            # CO KL cost per timestep
            self.kl_cost_co_b_t = KLCost_GaussianGaussian(self.cos_posterior,
                                                          self.cos_prior).kl_cost_b
            # CO KL cost for the batch
            self.kl_cost_co = tf.reduce_mean(
                tf.reduce_mean(self.kl_cost_co_b_t, [1]) )
            self.kl_cost += self.kl_cost_co * self.kl_co_weight


        ## calculate reconstruction cost
        # get final mask for gradient blocking
        if hps['keep_ratio'] != 1.0:
            # let the gradients pass through on blocked nodes with some probability
            random_tensor = tf.convert_to_tensor(1. - self.cd_grad_passthru_prob)
            random_tensor += tf.random_uniform(tf.shape(coor_drop_binary_mask),
                                                       dtype=coor_drop_binary_mask.dtype)
            # pass thru some gradients
            tmp_binary_mask =  coor_drop_binary_mask * tf.floor(random_tensor)
            # exclude cv samples
            grad_binary_mask = self.cv_binary_mask_batch * (1. - tmp_binary_mask)
        else:
            grad_binary_mask = self.cv_binary_mask_batch

        # block gradients for coordinated dropout and cross-validation
        if hps.output_dist.lower() == 'poisson':
            masked_logrates = entry_stop_gradients(self.logrates, grad_binary_mask)
            #self.loglikelihood_b_t = Poisson(masked_logrates).logp(self.dataset_ph)
            self.loglikelihood_b_t = -tf.nn.log_poisson_loss( self.dataset_ph, masked_logrates, compute_full_loss=True )
        elif hps.output_dist.lower() == 'gaussian':
            masked_output_mean = entry_stop_gradients(self.output_mean, grad_binary_mask)
            masked_output_logvar = entry_stop_gradients(self.output_logvar, grad_binary_mask)
            self.loglikelihood_b_t = diag_gaussian_log_likelihood(self.dataset_ph,
                                                                  masked_output_mean, masked_output_logvar)
            
        # cost for each trial

        #self.log_p_b_train =  (1. / self.cv_keep_ratio)**2 * tf.reduce_mean(
        #    tf.reduce_mean(self.loglikelihood_b_t * self.cv_binary_mask_batch, [2] ), [1] )
        # total rec cost (avg over all trials in batch)
        #self.log_p_train = tf.reduce_mean( self.log_p_b_train, [0])
        #self.rec_cost_train = -self.log_p_train
        #ones_ratio = tf.reduce_mean(self.cv_binary_mask_batch)

        self.rec_cost_train = - (1. / self.cv_keep_ratio) * \
                              tf.reduce_sum(self.loglikelihood_b_t * self.cv_binary_mask_batch) / tf.cast(graph_batch_size, tf.float32)
        #self.rec_cost_train = - tf.reduce_sum(self.loglikelihood_b_t * self.cv_binary_mask_batch) / graph_batch_size
        #self.rec_cost_train /= ones_ratio


        # Valid cost for each trial
        #self.log_p_b_valid = (1. / (1. - self.cv_keep_ratio))**2 * tf.reduce_mean(
        #    tf.reduce_mean(self.loglikelihood_b_t * (1. - self.cv_binary_mask_batch), [2] ), [1] )
        # total rec cost (avg over all trials in batch)
        #self.log_p_valid = tf.reduce_mean( self.log_p_b_valid, [0])
        #self.rec_cost_valid = -self.log_p_valid
        if self.cv_keep_ratio != 1.0:
            self.rec_cost_valid = - (1. / (1. - self.cv_keep_ratio)) * \
                                  tf.reduce_sum(self.loglikelihood_b_t * (1. - self.cv_binary_mask_batch)) / tf.cast(graph_batch_size, tf.float32)
        else:
            self.rec_cost_valid = tf.constant(np.nan)
        #self.rec_cost_valid = - tf.reduce_sum(self.loglikelihood_b_t * (1. - self.cv_binary_mask_batch)) / graph_batch_size
        #self.rec_cost_valid /= (1. - ones_ratio)



        # calculate L2 costs for each network
        # normalized by number of parameters.
        self.l2_cost = tf.constant(0.0)
        l2_costs = []
        l2_numels = []
        l2_reg_var_lists = ['l2_gen',
                            'l2_con',
                            'l2_ic_enc',
                            'l2_ci_enc',
                            'l2_gen_2_factors',
                            'l2_ci_enc_2_co_in',
                            ]

        l2_reg_scales = [hps.l2_gen_scale, hps.l2_con_scale,
                         hps.l2_ic_enc_scale, hps.l2_ci_enc_scale,
                         hps.l2_gen_2_factors_scale,
                         hps.l2_ci_enc_2_co_in]
        for l2_reg, l2_scale in zip(l2_reg_var_lists, l2_reg_scales):
            if l2_scale == 0:
                continue
            l2_reg_vars = tf.get_collection(l2_reg)

            for v in l2_reg_vars:
                numel = tf.reduce_prod(tf.concat(axis=0, values=tf.shape(v)))
                numel_f = tf.cast(numel, tf.float32)
                l2_numels.append(numel_f)
                v_l2 = tf.reduce_sum(v*v)
                l2_costs.append(0.5 * l2_scale * v_l2)
                #l2_costs.append(tf.nn.l2_loss(v))
        if l2_numels:
            self.l2_cost = tf.add_n(l2_costs) / tf.add_n(l2_numels)

        ## calculate total training cost
        self.total_cost = self.l2_weight * self.l2_cost + self.kl_weight * self.kl_cost + self.rec_cost_train
        

        if hps.do_train_encoder_only:
            # get the list of ci_enc and ic_enc variables
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='LFADS/ic_enc*')  + \
                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='LFADS/ci_enc*')
        else:
            # get the list of trainable variables
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            if hps.do_train_readin == False:
                # filter out any variables with name containing '_in_fac_linear'
                regex = re.compile('.+_in_fac_linear.+')
                trainable_vars = [i for i in trainable_vars if not regex.search(i.name)]

        self.trainable_vars = trainable_vars
        
        self.gradients = tf.gradients(self.total_cost, self.trainable_vars)
        self.gradients, self.grad_global_norm = \
                                                tf.clip_by_global_norm(
                                                    self.gradients, \
                                                    hps['max_grad_norm'])
        # this is the optimizer
        self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999,
                                     epsilon=1e-01)

        # global that holds current step number
        self.train_step = tf.get_variable("global_step", [], tf.int64,
                                     tf.zeros_initializer(),
                                     trainable=False)
        self.train_op = self.opt.apply_gradients(
            zip(self.gradients, self.trainable_vars), global_step = self.train_step)

        # hooks to save down model checkpoints:
        # "save every so often" (i.e., recent checkpoints)
        self.seso_saver = tf.train.Saver(tf.global_variables(),
                                     max_to_keep=hps.max_ckpt_to_keep)

        # lowest validation error checkpoint
        self.lve_saver = tf.train.Saver(tf.global_variables(),
                                    max_to_keep=hps.max_ckpt_to_keep)
        
        # store the hps
        self.hps = hps


        # Don't print this?
        '''
        print("Model Variables (to be optimized): ")
        total_params = 0
        tvars = self.trainable_vars
        for i in range(len(tvars)):
            shape = tvars[i].get_shape().as_list()
            print("- ", i, tvars[i].name, shape)
            total_params += np.prod(shape)
        print("Total model parameters: ", total_params)
        '''
        
        self.merged_generic = tf.summary.merge_all() # default key is 'summaries'
        session = tf.get_default_session()
        self.logfile = os.path.join(hps.lfads_save_dir, "lfads_log")
        self.writer = tf.summary.FileWriter(self.logfile, session.graph)

        
    ## functions to interface with the outside world
    def build_feed_dict(self, train_name, data_bxtxd, cv_rand_mask=None, ext_input_bxtxi=None, run_type=None,
                        keep_prob=None, kl_ic_weight=1.0, kl_co_weight=1.0,
                        keep_ratio=None, kl_weight=1.0, l2_weight=1.0):
      """Build the feed dictionary, handles cases where there is no value defined.

      Args:
        train_name: The key into the datasets, to set the tf.case statement for
          the proper readin / readout matrices.
        data_bxtxd: The data tensor
        keep_prob: The drop out keep probability.

      Returns:
        The feed dictionary with TF tensors as keys and data as values, for use
        with tf.Session.run()

      """
      # CP: the following elements must be defined in a feed_dict for the graph to run
      # (each is a placeholder in the graph)
      #   self.dataName
      #   self.dataset_ph
      #   self.kl_ic_weight
      #   self.kl_co_weight
      #   self.run_type
      #   self.keep_prob

      feed_dict = {}
      B, T, _ = data_bxtxd.shape
      feed_dict[self.dataName] = train_name
      feed_dict[self.dataset_ph] = data_bxtxd
      feed_dict[self.kl_ic_weight] = kl_ic_weight
      feed_dict[self.kl_co_weight] = kl_co_weight
      feed_dict[self.kl_weight] = kl_weight
      feed_dict[self.l2_weight] = l2_weight

      if self.ext_input is not None and ext_input_bxtxi is not None:
          feed_dict[self.ext_input] = ext_input_bxtxi

      if cv_rand_mask is None:
          feed_dict[self.cv_rand_mask] = np.ones_like(data_bxtxd)
      else:
          feed_dict[self.cv_rand_mask] = cv_rand_mask

      if run_type is None:
        feed_dict[self.run_type] = self.hps.kind
      else:
        feed_dict[self.run_type] = run_type
          
      if keep_prob is None:
        feed_dict[self.keep_prob] = self.hps.keep_prob
      else:
        feed_dict[self.keep_prob] = keep_prob

      if keep_ratio is None:
        feed_dict[self.keep_ratio] = self.hps.keep_ratio
      else:
        feed_dict[self.keep_ratio] = keep_ratio

      return feed_dict



    def shuffle_and_flatten_datasets(self, datasets, kind='train'):
      """Since LFADS supports multiple datasets in the same dynamical model,
      we have to be careful to use all the data in a single training epoch.  But
      since the datasets my have different data dimensionality, we cannot batch
      examples from data dictionaries together.  Instead, we generate random
      batches within each data dictionary, and then randomize these batches
      while holding onto the dataname, so that when it's time to feed
      the graph, the correct in/out matrices can be selected, per batch.

      Args:
        datasets: A dict of data dicts.  The dataset dict is simply a
          name(string)-> data dictionary mapping (See top of lfads.py).
        kind: 'train' or 'valid'

      Returns:
        A flat list, in which each element is a pair ('name', indices).
      """
      batch_size = self.hps.batch_size
      ndatasets = len(datasets)
      random_example_idxs = {}
      epoch_idxs = {}
      all_name_example_idx_pairs = []
      kind_data = kind + '_data'
      for name, data_dict in datasets.items():
        nexamples, ntime, data_dim = data_dict[kind_data].shape
        epoch_idxs[name] = 0
        if kind == 'valid':
            # pass one large batch for validation set
            random_example_idxs = [np.arange(nexamples)]
        else:
            random_example_idxs = \
                ListOfRandomBatches(nexamples, batch_size)

        epoch_size = len(random_example_idxs)
        names = [name] * epoch_size
        all_name_example_idx_pairs += zip(names, random_example_idxs)

      # shuffle the batches so the dataset order is scrambled
      np.random.shuffle(all_name_example_idx_pairs) #( shuffle in place)

      return all_name_example_idx_pairs


    def train_epoch(self, datasets, do_save_ckpt, kl_ic_weight, kl_co_weight, kl_weight, l2_weight):
    # train_epoch runs the entire training set once
    #    (it is mostly a wrapper around "run_epoch")
    # afterwards it saves a checkpoint if requested
        collected_op_values = self.run_epoch(datasets, kl_ic_weight,
                                             kl_co_weight, dataset_type="train",
                                             run_type = "train",
                                             kl_weight=kl_weight,
                                             l2_weight=l2_weight)

        if do_save_ckpt:
          session = tf.get_default_session()
          checkpoint_path = os.path.join(self.hps.lfads_save_dir,
                                         self.hps.checkpoint_name + '.ckpt')
          self.seso_saver.save(session, checkpoint_path,
                               global_step=self.train_step)

        return collected_op_values

    def do_validation(self, datasets, kl_ic_weight, kl_co_weight, dataset_type, kl_weight, l2_weight):
    # do_validation performs an evaluation of the reconstruction cost
    #    can do this on either train or valid datasets
    #    (it is mostly a wrapper around "run_epoch")
        collected_op_values = self.run_epoch(datasets, kl_ic_weight,
                                             kl_co_weight,  dataset_type=dataset_type,
                                             run_type = "valid",
                                             kl_weight=kl_weight,
                                             l2_weight=l2_weight
                                             )
        return collected_op_values

    def run_epoch(self, datasets, kl_ic_weight, kl_co_weight, dataset_type = "train", run_type="train",
                  kl_weight=1.0, l2_weight=1.0):
        ops_to_eval = [self.total_cost, self.rec_cost_train, self.rec_cost_valid,
                       self.kl_cost, self.l2_cost]
        # get a full list of all data for this type (train/valid)
        all_name_example_idx_pairs = \
          self.shuffle_and_flatten_datasets(datasets, dataset_type)
        
        if dataset_type == "train":
            kind_data = "train_data"
            cv_mask_name = "train_data_cvmask"
            ext_input_kind = "train_ext_input"

        else:
            kind_data = "valid_data"
            cv_mask_name = "valid_data_cvmask"
            ext_input_kind = "valid_ext_input"

        if run_type == "train":
            ops_to_eval.append(self.train_op)
            keep_prob = self.hps.keep_prob
            keep_ratio = self.hps.keep_ratio
        else:
            keep_prob = 1.0
            keep_ratio = 1.0
            
        session = tf.get_default_session()

        evald_ops = []
        # iterate over all datasets
        for name, example_idxs in all_name_example_idx_pairs:
            data_dict = datasets[name]
            data_extxd = data_dict[kind_data]
            cv_rand_mask = data_dict[cv_mask_name]
            ext_input_bxtxi = data_dict[ext_input_kind]

            this_batch = data_extxd[example_idxs,:,:]

            this_batch_cvmask = cv_rand_mask[example_idxs,:,:] if cv_rand_mask is not None else None

            ext_input_batch = ext_input_bxtxi[example_idxs, :, :] if ext_input_bxtxi is not None else None

            feed_dict = self.build_feed_dict(name, this_batch,
                                             cv_rand_mask=this_batch_cvmask,
                                             ext_input_bxtxi=ext_input_batch,
                                             keep_prob=keep_prob,
                                             run_type = kind_dict("train"),
                                             kl_ic_weight = kl_ic_weight,
                                             kl_co_weight = kl_co_weight,
                                             keep_ratio=keep_ratio,
                                             kl_weight=kl_weight,
                                             l2_weight=l2_weight)
            evald_ops_this_batch = session.run(ops_to_eval, feed_dict = feed_dict)
            # for training runs, there is an extra output argument. kill it
            if len(evald_ops_this_batch) > 5:
                tc, rc, rc_v, kl, l2, _= evald_ops_this_batch
                evald_ops_this_batch = (tc, rc, rc_v, kl, l2)
            evald_ops.append(evald_ops_this_batch)
        evald_ops = np.mean(evald_ops, axis=0)
        return evald_ops
        
    # MRK the following functions are not used at all
    # def train_batch(self, dict_from_py):
    #     session = tf.get_default_session()
    #     ops_to_eval = [self.train_op, self.total_cost, self.rec_cost_train, \
    #                          self.kl_cost, self.output_dist_params, self.learning_rate]
    #     feed_dict = {self.dataset_ph: dict_from_py['dataset_ph'],
    #                  self.keep_prob: dict_from_py['keep_prob'],
    #                  self.keep_ratio: dict_from_py['keep_ratio'],
    #                  self.run_type: kind_dict("train")}
    #     return session.run(ops_to_eval, feed_dict)
    #
    #
    # def validation_batch(self, dict_from_py):
    #     session = tf.get_default_session()
    #     ops_to_eval = [self.total_cost, self.rec_cost_train, self.kl_cost]
    #     feed_dict = {self.input_data: dict_from_py['input_data'],
    #                  self.keep_prob: 1.0, # don't need to lower keep_prob from validation
    #                  self.keep_ratio: 1.0,
    #                  self.run_type: kind_dict("train")}
    #     return session.run(ops_to_eval, feed_dict)


    def run_learning_rate_decay_opt(self):
    # decay the learning rate 
        session = tf.get_default_session()
        session.run(self.learning_rate_decay_op)

    def get_learning_rate(self):
    # return the current learning rate
        session = tf.get_default_session()
        return session.run(self.learning_rate)

    def get_kl_l2_weights(self, session):
    # MRK, get the KL and L2 ramp weights
        train_step = session.run(self.train_step)
        l2_weight = float(train_step - self.hps['l2_start_step']) / float(self.hps['l2_increase_steps'])
        # clip to 0-1
        l2_weight = min(max(l2_weight, 0), 1)

        kl_weight = float(train_step - self.hps['kl_start_step']) / float(self.hps['kl_increase_steps'])
        # clip to 0-1
        kl_weight = min(max(kl_weight, 0.0), 1.0)

        return kl_weight, l2_weight


    def train_model(self, datasets, target_num_epochs=None):
    # this is the main loop for training a model
        hps = self.hps

        # check to see if there are any validation datasets
        has_any_valid_set = False
        for data_dict in datasets.values():
          if data_dict['valid_data'] is not None:
            has_any_valid_set = True
            break

        session = tf.get_default_session()

        # monitor the learning rate
        lr = self.get_learning_rate()
        lr_stop = hps.learning_rate_stop

        # epoch counter
        nepoch = 0
        lve_epoch = 0

        num_save_checkpoint = 1     # save checkpoints every num_save_checkpoint steps

        # TODO: modify to work with multiple datasets
        name=datasets.keys()[0]
        
        data_dict = datasets[name]
        #data_extxd = data_dict["train_data"]
        #his_batch = data_extxd[0:hps['batch_size'],:,:]

        valid_costs = []

        kl_weight, l2_weight = self.get_kl_l2_weights(session)
        # print validation costs before the first training step
        val_total_cost, valid_set_heldin_samp_cost, valid_set_heldout_samp_cost, val_kl_cost, val_l2_cost = \
            self.do_validation(datasets,
                             kl_ic_weight=hps['kl_ic_weight'],
                             kl_co_weight=hps['kl_co_weight'],
                             dataset_type="train",
                             kl_weight=kl_weight,
                             l2_weight=l2_weight)
        train_step = session.run(self.train_step)
        self.printlog("Epoch:%d, step:%d (TRAIN, VALID): total: %.2f, %.2f\
        recon: %.2f, %.2f, %.2f,    kl: %.2f, %.2f, kl weight: %.2f" % \
              (-1, train_step, 0, val_total_cost,
               0, valid_set_heldin_samp_cost, valid_set_heldout_samp_cost, 0, val_kl_cost,
               kl_weight))
        # pre-load the lve checkpoint (used in case of loaded checkpoint)

        if hps['checkpoint_pb_load_name'] == 'checkpoint_lve':
            self.lve = valid_set_heldin_samp_cost
        else:
            self.lve = np.inf
        self.trial_recon_cost = valid_set_heldin_samp_cost
        self.samp_recon_cost = valid_set_heldout_samp_cost

        coef = 1.0 # smoothing coefficient - lower values mean more smoothing

        # calculate R^2 if true data is available
        do_r2_calc = data_dict['train_truth'] is not None

        while True:
            new_lr = self.get_learning_rate()
            #self.printlog('Starting learning rate: ', new_lr)
            # should we stop?
            if target_num_epochs is None:
                if new_lr < hps['learning_rate_stop']:
                    self.printlog("Learning rate criteria met")
                    break
            else:
                if nepoch == target_num_epochs:  # nepoch starts at 0
                    self.printlog("Num epoch criteria met. "
                          "Completed {} epochs.".format(nepoch))
                    break

            do_save_ckpt = True if nepoch % num_save_checkpoint == 0 else False
            # always save checkpoint for the last epoch
            if target_num_epochs is not None:
                do_save_ckpt = True if nepoch == (target_num_epochs-1) else do_save_ckpt

            start_time = time.time()

            # MRK, get the KL and L2 ramp weights
            kl_weight, l2_weight = self.get_kl_l2_weights(session)

            # CP/MRK: we no longer use these step-specific outputs
            #tr_total_cost, train_set_heldin_samp_cost, train_set_heldout_samp_cost, tr_kl_cost = \
            self.train_epoch(datasets, do_save_ckpt=do_save_ckpt,
                                 kl_ic_weight = hps['kl_ic_weight'],
                                 kl_co_weight = hps['kl_co_weight'],
                                 kl_weight=kl_weight,
                                 l2_weight=l2_weight)
            
            tr_total_cost, train_set_heldin_samp_cost, train_set_heldout_samp_cost, tr_kl_cost, l2_cost = \
                self.do_validation(datasets,
                                 kl_ic_weight = hps['kl_ic_weight'],
                                 kl_co_weight = hps['kl_co_weight'],
                                 dataset_type="train",
                                 kl_weight=kl_weight,
                                 l2_weight=l2_weight)

            val_total_cost, valid_set_heldin_samp_cost, valid_set_heldout_samp_cost, val_kl_cost, _  = \
                self.do_validation(datasets,
                                 kl_ic_weight = hps['kl_ic_weight'],
                                 kl_co_weight = hps['kl_co_weight'],
                                 dataset_type="valid",
                                 kl_weight=kl_weight,
                                 l2_weight=l2_weight)

            epoch_time = time.time() - start_time
            self.printlog("Elapsed time: %f" %epoch_time)

            if np.isnan(tr_total_cost) or np.isnan(val_total_cost):
                self.printlog('Nan found in training or validation cost evaluation. Training stopped!')
                self.trial_recon_cost = np.nan
                self.samp_recon_cost = np.nan
                break

            # Evaluate the model with posterior mean sampling
            #'''
            # every 10 epochs
            if do_r2_calc and (nepoch % 10 == 0):
                all_valid_R2_heldin = []
                all_valid_R2_heldout = []
                all_train_R2_heldin = []
                all_train_R2_heldout = []
                for data_name, data_dict in datasets.items():
                    # Validation R^2
                    data_kind, data_extxd = ('valid', data_dict['valid_data'])
                    ext_input_extxd = data_dict['valid_ext_input']
                    model_runs = self.eval_model_runs_avg_epoch(data_name, data_extxd, ext_input_extxd, pm_batch_size=1,
                                                                do_average_batch=True)
                    lfads_output = model_runs['output_dist_params']
                    if hps.output_dist.lower() == 'poisson':
                        lfads_output = lfads_output
                    elif hps.output_dist.lower() == 'gaussian':
                        raise NameError("Not implemented!")
                    # Get R^2
                    data_true = data_dict['valid_truth']
                    data_cvmask = data_dict['valid_data_cvmask']
                    heldin, heldout = self.get_R2(data_true, lfads_output, data_cvmask)
                    all_valid_R2_heldin.append(heldin)
                    all_valid_R2_heldout.append(heldout)

                    # training R^2
                    data_kind, data_extxd = ('train', data_dict['train_data'])
                    ext_input_extxd = data_dict['train_ext_input']
                    model_runs = self.eval_model_runs_avg_epoch(data_name, data_extxd, ext_input_extxd, pm_batch_size=1,
                                                                do_average_batch=True)
                    lfads_output = model_runs['output_dist_params']
                    if hps.output_dist.lower() == 'poisson':
                        lfads_output = lfads_output
                    elif hps.output_dist.lower() == 'gaussian':
                        raise NameError("Not implemented!")
                    # Get R^2
                    data_true = data_dict['train_truth']
                    data_cvmask = data_dict['train_data_cvmask']
                    heldin, heldout = self.get_R2(data_true, lfads_output, data_cvmask)
                    all_train_R2_heldin.append(heldin)
                    all_train_R2_heldout.append(heldout)

                all_valid_R2_heldin = np.mean(all_valid_R2_heldin)
                all_valid_R2_heldout = np.mean(all_valid_R2_heldout)

                all_train_R2_heldin = np.mean(all_train_R2_heldin)
                all_train_R2_heldout = np.mean(all_train_R2_heldout)
            else:
                all_valid_R2_heldin = np.nan
                all_valid_R2_heldout = np.nan
                all_train_R2_heldin = np.nan
                all_train_R2_heldout = np.nan

            # initialize the running average
            if nepoch == 0:
                smth_trial_val_recn_cost = valid_set_heldin_samp_cost
                smth_samp_val_recn_cost = train_set_heldout_samp_cost

            # recon cost over validation trials
            smth_trial_val_recn_cost = (1. - coef) * smth_trial_val_recn_cost + coef * valid_set_heldin_samp_cost
            # recon cost over dropped samples
            smth_samp_val_recn_cost = (1. - coef) * smth_samp_val_recn_cost + coef * train_set_heldout_samp_cost

            train_step = session.run(self.train_step)
            if np.isnan(all_train_R2_heldin):
                self.printlog("Epoch:%d, step:%d (TRA,VAL_SAMP,VAL_TRI): tot:%.2f,%.2f, rec:%.2f,%.2f,%.2f, kl:%.2f,%.2f, l2:%.4f, kl_weight:%.2f, l2_weight:%.2f" % \
                      (nepoch, train_step, tr_total_cost, val_total_cost,
                       train_set_heldin_samp_cost, train_set_heldout_samp_cost, valid_set_heldin_samp_cost,
                       tr_kl_cost, val_kl_cost, l2_cost, kl_weight, l2_weight))
            else:
                self.printlog("Epoch:%d, step:%d (TRA,VAL_SAMP,VAL_TRI): tot:%.2f,%.2f, rec:%.2f,%.2f,%.2f, kl:%.2f,%.2f, l2:%.4f, kl_weight:%.2f, l2_weight:%.2f, R^2(T/V:Held-in,Held-out),%.3f,%.3f,%.3f,%.3f" % \
                      (nepoch, train_step, tr_total_cost, val_total_cost,
                       train_set_heldin_samp_cost, train_set_heldout_samp_cost, valid_set_heldin_samp_cost,
                       tr_kl_cost, val_kl_cost, l2_cost, kl_weight, l2_weight,
                       all_train_R2_heldin, all_train_R2_heldout, all_valid_R2_heldin, all_valid_R2_heldout,))

            is_lve = smth_trial_val_recn_cost < self.lve

            # Making parameters available for lfads_wrappper
            if hps['checkpoint_pb_load_name'] == 'checkpoint_lve':
                # if we are returning lve costs
                if is_lve:
                    self.trial_recon_cost = smth_trial_val_recn_cost
                    self.samp_recon_cost = smth_samp_val_recn_cost
            else:
                self.trial_recon_cost = smth_trial_val_recn_cost
                self.samp_recon_cost = smth_samp_val_recn_cost

            # MRK, moved this here to get the right for the checkpoint train_step
            if is_lve:
                # new lowest validation error
                self.lve = smth_trial_val_recn_cost
                # MRK, make lve accessible from the model class
                #self.lve = lve
                lve_epoch = nepoch
                checkpoint_path = os.path.join(self.hps.lfads_save_dir,
                                               self.hps.checkpoint_name + '_lve.ckpt')
                # MRK, for convenience, it can be reconstructed from the paths in hps
                #self.lve_checkpoint = checkpoint_path

                self.lve_saver.save(session, checkpoint_path,
                                    global_step=self.train_step,
                                    latest_filename='checkpoint_lve')

            # TODO: write tensorboard summaries here...
            
            # write csv log file
            if self.hps.csv_log:
                # construct an output string
                csv_outstr = "epoch,%d, step,%d, total,%.6E,%.6E, \
                recon,%.6E,%.6E,%.6E,%.6E, R^2 (Held-in, Held-out), %.6E, %.6E, %.6E, %.6E," \
                             "kl,%.6E,%.6E, l2,%.6E, klweight,%.6E, l2weight,%.6E\n"% \
                (nepoch, train_step, tr_total_cost, val_total_cost,
                 train_set_heldin_samp_cost, train_set_heldout_samp_cost, valid_set_heldin_samp_cost, valid_set_heldout_samp_cost,
                 all_train_R2_heldin, all_train_R2_heldout, all_valid_R2_heldin, all_valid_R2_heldout,
                 tr_kl_cost, val_kl_cost, l2_cost, kl_weight, l2_weight)
                # log to file
                csv_file = os.path.join(self.hps.lfads_save_dir, self.hps.csv_log+'.csv')
                with open(csv_file, "a") as myfile:
                    myfile.write(csv_outstr)

            #plotind = random.randint(0, hps['batch_size']-1)
            #ops_to_eval = [self.output_dist_params]
            #output = session.run(ops_to_eval, feed_dict)
            #plt = plot_data(this_batch[plotind,:,:], output[0][plotind,:,:])
            #if nepoch % 15 == 0:
            #    close_all_plots()

            # should we decrement learning rate?
            # MRK, for PBT we can set n_lr to np.inf
            n_lr = int(hps['learning_rate_n_to_compare'])

            # MRK, change the LR decay based on valid cost (previously was based on train cost)
            #valid_cost_to_use = val_total_cost
            # use training cost for learning rate annealing
            valid_cost_to_use = val_total_cost
            #if n_lr > 0 and len(valid_costs) > n_lr and (valid_cost_to_use > max(valid_costs[-n_lr:])):
            #    self.printlog("Decreasing learning rate")
            #    self.run_learning_rate_decay_opt()
            if n_lr > 0 and len(valid_costs) > n_lr and (valid_cost_to_use > max(valid_costs[-n_lr:])):
                self.run_learning_rate_decay_opt()
                lr = session.run( self.learning_rate )
                self.printlog("Decreasing learning rate to ", lr)
                valid_costs.append(np.inf)

            # early stopping when no improvement of validation cost for 3*n_lr
            #if len(valid_costs) > n_lr*3 and valid_cost_to_use > np.max(valid_costs[-n_lr*3:]):
            if n_lr > 0 and nepoch - lve_epoch > 3*n_lr:
                self.printlog("No improvement on the validation cost! Stopping the training!")
                break

            valid_costs.append(valid_cost_to_use)
            #train_costs.append(tr_total_cost)

            nepoch += 1


    def eval_model_runs_batch(self, data_name, data_bxtxd, ext_input_bxtxi,
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

        feed_dict = self.build_feed_dict(data_name, data_bxtxd, cv_rand_mask=np.ones_like(data_bxtxd),
        	ext_input_bxtxi=ext_input_bxtxi, run_type=run_type,
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
    def eval_model_runs_avg_epoch(self, data_name, data_extxd, ext_input_bxtxi, pm_batch_size=None,
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
          self.printlog("Setting number of posterior samples to process to : %d" % E)
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
            mv = self.eval_model_runs_batch(data_name, data_bxtxd, ext_input_bxtxi,
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

        self.printlog("")
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

    def get_R2(self, data_true, data_est, mask):
        """Calculate the R^2 between the true rates and LFADS output, over all
        trials and all channels
        """
        true_flat = data_true.flatten()
        est_flat = data_est.flatten()
        if mask is not None:
            mask = mask.flatten()
            mask = mask.astype(np.bool)
            R2_heldin = np.corrcoef(true_flat[mask], est_flat[mask])**2.0
            R2_heldout = np.corrcoef(true_flat[np.invert(mask)], est_flat[np.invert(mask)])**2.0
            return R2_heldin[0,1], R2_heldout[0,1]
        else:
            R2_heldin = np.corrcoef(true_flat, est_flat) ** 2.0
            return R2_heldin[0, 1], np.nan

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
          data_tuple = [('train', data_dict['train_data'], data_dict['train_ext_input']),
                        ('valid', data_dict['valid_data'], data_dict['valid_ext_input'])]
          for data_kind, data_extxd, ext_input_bxtxi in data_tuple:
            if not output_fname:
              fname = "model_runs_" + data_name + '_' + data_kind + '_' + kind
            else:
              fname = output_fname + data_name + '_' + data_kind + '_' + kind

            self.printlog("Writing data for %s data and kind %s to file %s." % (data_name, data_kind, fname))
            model_runs = self.eval_model_runs_avg_epoch(data_name, data_extxd, ext_input_bxtxi)
            all_model_runs.append(model_runs)
            full_fname = os.path.join(hps.lfads_save_dir, fname)
            write_data(full_fname, model_runs, compression='gzip')
            self.printlog("Done.")

        return all_model_runs

            
