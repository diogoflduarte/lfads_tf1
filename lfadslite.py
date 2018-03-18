import tensorflow as tf
import numpy as np
import sys
import time
import os
import random
import matplotlib.pyplot as plt


# utils defined by CP
from helper_funcs import linear, init_linear_transform, makeInitialState
from helper_funcs import ListOfRandomBatches, kind_dict, kind_dict_key
from helper_funcs import DiagonalGaussianFromInput, DiagonalGaussian, DiagonalGaussianFromExisting, diag_gaussian_log_likelihood
from helper_funcs import BidirectionalDynamicRNN, DynamicRNN, LinearTimeVarying
from helper_funcs import KLCost_GaussianGaussian, Poisson
from helper_funcs import write_data
from helper_funcs import printer
from plot_funcs import plot_data, close_all_plots
from data_funcs import read_datasets
from customcells import ComplexCell
from customcells import CustomGRUCell
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


class LFADS(object):

    def __init__(self, hps, datasets = None):

        # to stop certain gradients paths through the graph in backprop
        def entry_stop_gradients(target, mask):
            mask_h = tf.abs(1 - mask)
            return tf.stop_gradient(mask_h * target) + mask * target

    # build the graph
        print("This is lfadslite with L2")
        # set the learning rate, defaults to the hyperparam setting
        # TODO: test if this properly re-initializes learning rate when loading a new model
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
            # dropout keep probability
            #   enumerated in helper_funcs.kind_dict
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.keep_ratio = tf.placeholder(tf.float32, name='keep_ratio')

            self.run_type = tf.placeholder(tf.int16, name='run_type')
            self.kl_ic_weight = tf.placeholder(tf.float32, name='kl_ic_weight')
            self.kl_co_weight = tf.placeholder(tf.float32, name='kl_co_weight')
            # name of the dataset
            self.dataName = tf.placeholder(tf.string, shape=(), name='dataset_name')

         
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
        self.random_masks_np = random_masks_np = [None] * ndatasets
        self.fns_cv_binary_masks = fns_cv_binary_masks = [None] * ndatasets
        # ext_inputs are not yet implemented, but, placeholder for later
        self.ext_inputs = ext_inputs = None
        
        # figure out the input (dataset) dimensionality
        #allsets = hps['dataset_dims'].keys()
        #self.input_dim = hps['dataset_dims'][allsets[0]]
        
        self.cv_keep_ratio = hps['cv_keep_ratio']

        ## do per-session stuff
        for d, name in enumerate( dataset_names ):
            data_dim = hps.dataset_dims[name]

            # Step 0) define the preds comparator for this dataset
            preds[ d ] = tf.equal( tf.constant( name ), self.dataName )
            
            # Step 1) alignment matrix stuff.
            # the alignment matrix only matters if in_factors_dim is nonzero
            if hps.in_factors_dim > 0:
                in_mat_cxf = None
                align_bias_1xc = None
                in_bias_1xf = None
            
                # get the alignment_matrix if provided
                if 'alignment_matrix_cxf' in datasets[ name ].keys():
                    in_mat_cxf = datasets[ name ][ 'alignment_matrix_cxf'].astype( np.float32 )
                    # check that sizing is appropriate
                    if in_mat_cxf.shape != (data_dim, hps.in_factors_dim):
                        raise ValueError("""Alignment matrix must have dimensions %d x %d
                        (data_dim x factors_dim), but currently has %d x %d."""%
                                         (data_dim, hps.infactors_dim, in_mat_cxf.shape[0],
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
            if hps.cv_rand_seed:
                np.random.seed( int(hps.cv_rand_seed) + d)

            # Step 2) make a cross-validation random mask for each dataset
            # uniform [cv_keep_ratio, 1.0 + cv_keep_ratio)
            random_mask_np_this_dataset = self.cv_keep_ratio + np.random.rand(hps['num_steps'], hps.dataset_dims[ name ])
            # convert to 0 and 1
            random_mask_np_this_dataset = np.floor(random_mask_np_this_dataset)
            random_masks_np[ d ] = random_mask_np_this_dataset
            # converting to tensor
            fns_cv_binary_masks[ d ] = makelambda( tf.convert_to_tensor(self.random_masks_np[ d ], dtype=tf.float32) )

            #reset the np random seed to enforce randomness for the other random draws
            np.random.seed()
            
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
        pf_pairs_cv_binary_masks = zip(preds, fns_cv_binary_masks )

        # now, choose the ones for this session
        this_dataset_in_fac_W = _case_with_no_default( pf_pairs_in_fac_Ws )
        this_dataset_in_fac_b = _case_with_no_default( pf_pairs_in_fac_bs )
        this_dataset_out_fac_W = _case_with_no_default( pf_pairs_out_fac_Ws )
        this_dataset_out_fac_b = _case_with_no_default( pf_pairs_out_fac_bs )
        this_dataset_cv_binary_mask = _case_with_no_default( pf_pairs_cv_binary_masks )
                

        
        
        # batch_size - read from the data placeholder
        graph_batch_size = tf.shape(self.dataset_ph)[0]

        # can we infer the data dimensionality for the random mask?
        data_dim_tensor = tf.shape(self.dataset_ph)[2]
        dataset_dims = self.dataset_ph.get_shape()
        data_dim = dataset_dims[2]
        print(data_dim)
        
        # coordinated dropout
        if hps['keep_ratio'] != 1.0:
            # coordinated dropout enabled on inputs
            masked_dataset_ph, coor_drop_binary_mask = dropout(self.dataset_ph, self.keep_ratio)
        else:
            # no coordinated dropout
            masked_dataset_ph = self.dataset_ph

        # replicate the cross-validation binary mask for this dataset for all elements of the batch
        self.cv_binary_mask_batch = tf.expand_dims(tf.ones([graph_batch_size, 1]), 1) * \
                                    this_dataset_cv_binary_mask
        
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
            ic_enc_cell = CustomGRUCell(num_units = hps['ic_enc_dim'],\
                                        batch_size = graph_batch_size,
                                        clip_value = hps['cell_clip_value'],
                                        recurrent_collections=['l2_ic_enc'])
            seq_len = hps['num_steps']
            #seq_len.set_shape([1, graph_batch_size])
            ## ic_encoder
            self.ic_enc_rnn_obj = BidirectionalDynamicRNN(
                state_dim = hps['ic_enc_dim'],
                batch_size = graph_batch_size,
                name = 'ic_enc',
                sequence_lengths = seq_len,
                cell = ic_enc_cell,
                inputs = self.input_to_encoders,
                initial_state = None,
                rnn_type = 'customgru',
                output_keep_prob = self.keep_prob)
            #    inputs = masked_dataset_ph,

            # map the ic_encoder onto the actual ic layer
            self.gen_ics_posterior = DiagonalGaussianFromInput(
                x = self.ic_enc_rnn_obj.last_tot,
                z_size = hps['ic_dim'],
                name = 'ic_enc_2_ics',
                var_min = hps['ic_post_var_min'],
            )
            self.posterior_zs_g0 = self.gen_ics_posterior

        # to go forward, either sample from the posterior, or take mean
        #     (depending on current usage)
        if self.run_type in [kind_dict("train"), kind_dict("posterior_sample_and_average")]:
            self.gen_ics_lowd = self.gen_ics_posterior.sample()
        else:
            self.gen_ics_lowd = self.gen_ics_posterior.mean
        with tf.variable_scope('generator'):
            self.gen_ics = linear(self.gen_ics_lowd, hps['gen_dim'], name='ics_2_g0')


        ### CONTROLLER construction
        # this should only be done if a controller is requested
        # if not, skip all these graph elements like so:
        if hps['co_dim'] ==0:
            with tf.variable_scope('generator'):
                gen_cell = CustomGRUCell(num_units = hps['ic_enc_dim'],
                                         batch_size = graph_batch_size,
                                         clip_value = hps['cell_clip_value'],
                                         recurrent_collections=['l2_gen'])
                # setup generator
                self.gen_rnn_obj = DynamicRNN(state_dim = hps['gen_dim'],
                                              batch_size = graph_batch_size,
                                              name = 'gen',
                                              sequence_lengths = seq_len,
                                              cell = gen_cell,
                                              inputs = None,
                                              initial_state = self.gen_ics,
                                              rnn_type = 'customgru',
                                              output_keep_prob = self.keep_prob)
                self.gen_states = self.gen_rnn_obj.states

            with tf.variable_scope('factors'):

                ## factors
                self.fac_obj = LinearTimeVarying(inputs = self.gen_rnn_obj.states,
                                                    output_size = hps['factors_dim'],
                                                    transform_name = 'gen_2_factors',
                                                    output_name = 'factors_concat',
                                                    collections='l2_gen_2_factors'
                                                 )
                self.factors = self.fac_obj.output
        else:
            with tf.variable_scope('ci_enc'):

                ci_enc_cell = CustomGRUCell(num_units = hps['ci_enc_dim'],\
                                            batch_size = graph_batch_size,
                                            clip_value = hps['cell_clip_value'],
                                            recurrent_collections=['l2_ci_enc'])
                ## ci_encoder
                self.ci_enc_rnn_obj = BidirectionalDynamicRNN(
                    state_dim = hps['ci_enc_dim'],
                    batch_size = graph_batch_size,
                    name = 'ci_enc',
                    sequence_lengths = seq_len,
                    cell = ci_enc_cell,
                    inputs = self.input_to_encoders,
                    initial_state = None,
                    rnn_type = 'customgru',
                    output_keep_prob = self.keep_prob)
                #    inputs = masked_dataset_ph,

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
                with tf.variable_scope('ci_enc_2_co_in'):
                    # one input from the encoder, another will come back from factors
                    self.ci_enc_object = LinearTimeVarying(
                        inputs=self.ci_enc_rnn_states,
                        output_size = hps['con_ci_enc_in_dim'],
                        transform_name = 'ci_enc_2_co_in',
                        output_name = 'ci_enc_output_concat',
                        nonlinearity = None,
                        collections=['l2_ci_enc_2_co_in'])
                    self.ci_enc_outputs = self.ci_enc_object.output

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
                self.comcell_state_dims = [hps['con_dim'],
                                           hps['gen_dim'],
                                           hps['co_dim'], # for the controller output means
                                           hps['co_dim'], # for the variances
                                           hps['co_dim'], # for the sampled controller output
                                           hps['factors_dim']]

                # construct the complexcell
                self.complexcell=ComplexCell(hps['con_dim'],
                                             hps['gen_dim'],
                                             hps['co_dim'],
                                             hps['factors_dim'],
                                             hps['con_fac_in_dim'],
                                             graph_batch_size,
                                             var_min = hps['co_post_var_min'],
                                             kind = self.run_type)

                # construct the actual RNN
                #   its inputs are the output of the controller_input_enc
                self.complex_outputs, self.complex_final_state =\
                tf.nn.dynamic_rnn(self.complexcell,
                                  inputs = self.ci_enc_outputs,
                                  initial_state = self.complexcell_init_state,
                                  time_major=False)

                # split the states of the individual RNNs
                # from the packed "complexcell" state
                self.con_states, self.gen_states, self.co_mean_states, self.co_logvar_states, self.controller_outputs, self.factors =\
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
        self.gen_ics_prior = DiagonalGaussian(
            z_size = [graph_batch_size, hps['ic_dim']], name='gen_ics_prior',
            var = hps['ic_prior_var'])
        self.prior_zs_g0 = self.gen_ics_prior
        
        # g0 KL cost for each trial
        self.kl_cost_g0_b = KLCost_GaussianGaussian(self.gen_ics_posterior,
                                                    self.gen_ics_prior).kl_cost_b
        # g0 KL cost for the whole batch
        self.kl_cost_g0 = tf.reduce_mean(self.kl_cost_g0_b)
        # total KL cost
        self.kl_cost = self.kl_cost_g0 * self.kl_ic_weight
        if hps['co_dim'] !=0 :
            # if there are controller outputs, calculate a KL cost for them
            # first build a prior to compare to
            self.cos_prior = DiagonalGaussian(
                z_size = [graph_batch_size, hps['num_steps'], hps['co_dim']],
                name='cos_prior', var = hps['co_prior_var'])
            # then build a posterior
            self.cos_posterior = DiagonalGaussianFromExisting(
                self.co_mean_states,
                self.co_logvar_states)

            # CO KL cost per timestep
            self.kl_cost_co_b_t = KLCost_GaussianGaussian(self.cos_posterior,
                                                          self.cos_prior).kl_cost_b
            # CO KL cost for the batch
            self.kl_cost_co = tf.reduce_mean(
                tf.reduce_mean(self.kl_cost_co_b_t, [1]) )
            self.kl_cost += self.kl_cost_co * self.kl_co_weight


        # todo test gradient blocking for gaussian dist
        ## calculate reconstruction cost
        # get final mask for gradient blocking
        if hps['keep_ratio'] != 1.0:
            grad_binary_mask = self.cv_binary_mask_batch * (1. - coor_drop_binary_mask)
        else:
            grad_binary_mask = self.cv_binary_mask_batch

        # block gradients for coordinated dropout and cross-validation
        if hps.output_dist.lower() == 'poisson':
            masked_logrates = entry_stop_gradients(self.logrates, grad_binary_mask)
            self.loglikelihood_b_t = Poisson(masked_logrates).logp(self.dataset_ph)
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

        print(l2_reg_var_lists)
        l2_reg_scales = [hps.l2_gen_scale, hps.l2_con_scale,
                         hps.l2_ic_enc_scale, hps.l2_ci_enc_scale,
                         hps.l2_gen_2_factors_scale,
                         hps.l2_ci_enc_2_co_in]
        for l2_reg, l2_scale in zip(l2_reg_var_lists, l2_reg_scales):
            print(l2_scale)
            if l2_scale == 0:
                continue
            l2_reg_vars = tf.get_collection(l2_reg)
            for v in l2_reg_vars:
                numel = tf.reduce_prod(tf.concat(axis=0, values=tf.shape(v)))
                numel_f = tf.cast(numel, tf.float32)
                l2_numels.append(numel_f)
                v_l2 = tf.reduce_sum(v*v)
                l2_costs.append(0.5 * l2_scale * v_l2)
        print(l2_numels)
        if l2_numels:
            print("L2 cost applied")
            self.l2_cost = tf.add_n(l2_costs) / tf.add_n(l2_numels)

        ## calculate total training cost
        self.total_cost = self.l2_cost + self.kl_cost + self.rec_cost_train
        

        # get the list of trainable variables
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        #TODO - make sure input matrix / bias is not on list of trainable vars if that's not what we want
        if hps.do_train_readin==False:
            raise SyntaxError(' have not yet implemented the do_train_readin = False case... ')
        
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


        print("Model Variables (to be optimized): ")
        total_params = 0
        tvars = self.trainable_vars
        for i in range(len(tvars)):
            shape = tvars[i].get_shape().as_list()
            print("- ", i, tvars[i].name, shape)
            total_params += np.prod(shape)
        print("Total model parameters: ", total_params)

        
        self.merged_generic = tf.summary.merge_all() # default key is 'summaries'
        session = tf.get_default_session()
        self.logfile = os.path.join(hps.lfads_save_dir, "lfads_log")
        self.writer = tf.summary.FileWriter(self.logfile, session.graph)

        
    ## functions to interface with the outside world
    def build_feed_dict(self, train_name, data_bxtxd, run_type = None,
                        keep_prob=None, kl_ic_weight=1.0, kl_co_weight=1.0,
                        keep_ratio=None, cv_keep_ratio=None):
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
        random_example_idxs = \
          ListOfRandomBatches(nexamples, batch_size)

        epoch_size = len(random_example_idxs)
        names = [name] * epoch_size
        all_name_example_idx_pairs += zip(names, random_example_idxs)

      # shuffle the batches so the dataset order is scrambled
      np.random.shuffle(all_name_example_idx_pairs) #( shuffle in place)

      return all_name_example_idx_pairs


    def train_epoch(self, datasets, do_save_ckpt, kl_ic_weight, kl_co_weight):
    # train_epoch runs the entire training set once
    #    (it is mostly a wrapper around "run_epoch")
    # afterwards it saves a checkpoint if requested
        collected_op_values = self.run_epoch(datasets, kl_ic_weight,
                                             kl_co_weight, "train")

        if do_save_ckpt:
          session = tf.get_default_session()
          checkpoint_path = os.path.join(self.hps.lfads_save_dir,
                                         self.hps.checkpoint_name + '.ckpt')
          self.seso_saver.save(session, checkpoint_path,
                               global_step=self.train_step)

        return collected_op_values

    def valid_epoch(self, datasets, kl_ic_weight, kl_co_weight):
    # valid_epoch runs the entire validation set
    #    (it is mostly a wrapper around "run_epoch")
        collected_op_values = self.run_epoch(datasets, kl_ic_weight,
                                             kl_co_weight, "valid")
        return collected_op_values

    def run_epoch(self, datasets, kl_ic_weight, kl_co_weight, train_or_valid = "train"):
        ops_to_eval = [self.total_cost, self.rec_cost_train, self.rec_cost_valid,
                       self.kl_cost]
        # get a full list of all data for this type (train/valid)
        all_name_example_idx_pairs = \
          self.shuffle_and_flatten_datasets(datasets, train_or_valid)
        
        if train_or_valid == "train":
            ops_to_eval.append(self.train_op)
            kind_data = "train_data"
            keep_prob = self.hps.keep_prob
            keep_ratio = self.hps.keep_ratio

        else:
            kind_data = "valid_data"
            keep_prob = 1.0
            keep_ratio = 1.0
            
        session = tf.get_default_session()

        evald_ops = []
        # iterate over all datasets
        for name, example_idxs in all_name_example_idx_pairs:
            data_dict = datasets[name]
            data_extxd = data_dict[kind_data]

            this_batch = data_extxd[example_idxs,:,:]
            feed_dict = self.build_feed_dict(name, this_batch, keep_prob=keep_prob,
                                             run_type = kind_dict("train"),
                                             kl_ic_weight = kl_ic_weight,
                                             kl_co_weight = kl_co_weight,
                                             keep_ratio=keep_ratio)
            evald_ops_this_batch = session.run(ops_to_eval, feed_dict = feed_dict)
            # for training runs, there is an extra output argument. kill it
            if len(evald_ops_this_batch) > 4:
                tc, rc, rc_v, kl, _= evald_ops_this_batch
                evald_ops_this_batch = (tc, rc, rc_v, kl)
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

        num_save_checkpoint = 1     # save checkpoints every num_save_checkpoint steps

        # TODO: modify to work with multiple datasets
        name=datasets.keys()[0]
        
        data_dict = datasets[name]
        data_extxd = data_dict["train_data"]
        this_batch = data_extxd[0:hps['batch_size'],:,:]

        train_costs = []

        # print validation costs before the first training step
        val_total_cost, val_recon_cost, samp_val, val_kl_cost = \
            self.valid_epoch(datasets,
                             kl_ic_weight=hps['kl_ic_weight'],
                             kl_co_weight=hps['kl_co_weight'])
        kl_weight = hps['kl_ic_weight']
        train_step = session.run(self.train_step)
        print("Epoch:%d, step:%d (TRAIN, VALID): total: %.2f, %.2f\
        recon: %.2f, %.2f, %.2f,    kl: %.2f, %.2f, kl weight: %.2f" % \
              (-1, train_step, 0, val_total_cost,
               0, val_recon_cost, samp_val, 0, val_kl_cost,
               kl_weight))
        # pre-load the lve checkpoint (used in case of loaded checkpoint)
        self.lve = val_recon_cost
        self.trial_recon_cost = np.nan
        self.samp_recon_cost = np.nan
        coef = 0.9 # smoothing coefficient - lower values mean more smoothing

        while True:
            new_lr = self.get_learning_rate()
            # should we stop?
            if target_num_epochs is None:
                if new_lr < hps['learning_rate_stop']:
                    print("Learning rate criteria met")
                    break
            else:
                if nepoch == target_num_epochs:  # nepoch starts at 0
                    print("Num epoch criteria met. "
                          "Completed {} epochs.".format(nepoch))
                    break

            do_save_ckpt = True if nepoch % num_save_checkpoint == 0 else False
            # always save checkpoint for the last epoch
            if target_num_epochs is not None:
                do_save_ckpt = True if nepoch == (target_num_epochs-1) else do_save_ckpt

            start_time = time.time()
            tr_total_cost, tr_recon_cost_train, tr_recon_cost_valid, tr_kl_cost = \
                self.train_epoch(datasets, do_save_ckpt=do_save_ckpt,
                                 kl_ic_weight = hps['kl_ic_weight'],
                                 kl_co_weight = hps['kl_co_weight'])
            epoch_time = time.time() - start_time
            print("Elapsed time: %f" %epoch_time)

            val_total_cost, trial_val_recon_cost_train, trial_val_recon_cost_valid, val_kl_cost = \
                self.valid_epoch(datasets,
                                 kl_ic_weight = hps['kl_ic_weight'],
                                 kl_co_weight = hps['kl_co_weight'])

            if np.isnan(tr_total_cost) or np.isnan(val_total_cost):
                print('Nan found in training or validation cost evaluation. Training stopped!')
                break

            # Evaluate the model with posterior mean sampling
            '''
            all_valid_R2_heldin = []
            all_valid_R2_heldout = []
            all_train_R2_heldin = []
            all_train_R2_heldout = []
            for data_name, data_dict in datasets.items():
                # Validation R^2
                data_kind, data_extxd = ('valid', data_dict['valid_data'])
                model_runs = self.eval_model_runs_avg_epoch(data_name, data_extxd, 1,
                                                            do_average_batch=True)
                lfads_output = model_runs['output_dist_params']
                if hps.output_dist.lower() == 'poisson':
                    lfads_output = lfads_output
                elif hps.output_dist.lower() == 'gaussian':
                    raise NameError("Not implemented!")
                # Get R^2
                data_true = data_dict['valid_truth']
                heldin, heldout = self.get_R2(data_true, lfads_output, self.random_mask_np)
                all_valid_R2_heldin.append(heldin)
                all_valid_R2_heldout.append(heldout)

                # training R^2
                data_kind, data_extxd = ('train', data_dict['train_data'])
                model_runs = self.eval_model_runs_avg_epoch(data_name, data_extxd, 1,
                                                            do_average_batch=True)
                lfads_output = model_runs['output_dist_params']
                if hps.output_dist.lower() == 'poisson':
                    lfads_output = lfads_output
                elif hps.output_dist.lower() == 'gaussian':
                    raise NameError("Not implemented!")
                # Get R^2
                data_true = data_dict['train_truth']
                heldin, heldout = self.get_R2(data_true, lfads_output, self.random_mask_np)
                all_train_R2_heldin.append(heldin)
                all_train_R2_heldout.append(heldout)

            all_valid_R2_heldin = np.mean(all_valid_R2_heldin)
            all_valid_R2_heldout = np.mean(all_valid_R2_heldout)

            all_train_R2_heldin = np.mean(all_train_R2_heldin)
            all_train_R2_heldout = np.mean(all_train_R2_heldout)
            '''
            all_valid_R2_heldin = 0.
            all_valid_R2_heldout = 0.

            all_train_R2_heldin = 0.
            all_train_R2_heldout = 0.

            # initialize the running average
            if nepoch == 0:
                smth_trial_val_recn_cost = trial_val_recon_cost_train
                smth_samp_val_recn_cost = tr_recon_cost_valid

            # recon cost over validation trials
            smth_trial_val_recn_cost = (1. - coef) * smth_trial_val_recn_cost + coef * trial_val_recon_cost_train
            # recon cost over dropped samples
            smth_samp_val_recn_cost = (1. - coef) * smth_samp_val_recn_cost + coef * tr_recon_cost_valid

            kl_weight = hps['kl_ic_weight']
            train_step = session.run(self.train_step)
            print("Epoch:%d, step:%d (TRAIN, VALID_SAMP, VALID_TRIAL): total: %.4f, %.4f,\
            recon: %.4f, %.4f, %.4f, R^2 (T/V: Held-in, Held-out), %.3f, %.3f, %.3f, %.3f, kl: %.4f, %.4f, kl weight: %.4f" % \
                  (nepoch, train_step, tr_total_cost, val_total_cost,
                   tr_recon_cost_train, tr_recon_cost_valid, trial_val_recon_cost_train,
                   all_train_R2_heldin, all_train_R2_heldout, all_valid_R2_heldin, all_valid_R2_heldout,
                   tr_kl_cost, val_kl_cost,
                   kl_weight))

            self.trial_recon_cost = smth_trial_val_recn_cost
            self.samp_recon_cost = smth_samp_val_recn_cost
            # MRK, moved this here to get the right for the checkpoint train_step
            if smth_trial_val_recn_cost < self.lve:
                # new lowest validation error
                self.lve = smth_trial_val_recn_cost
                # MRK, make lve accessible from the model class
                #self.lve = lve

                checkpoint_path = os.path.join(self.hps.lfads_save_dir,
                                               self.hps.checkpoint_name + '_lve.ckpt')
                # MRK, for convenience, it can be reconstructed from the paths in hps
                #self.lve_checkpoint = checkpoint_path

                self.lve_saver.save(session, checkpoint_path,
                                    global_step=self.train_step,
                                    latest_filename='checkpoint_lve')

            #l2 has not been implemented yet
            l2_cost = 0
            l2_weight = 0

            # TODO: write tensorboard summaries here...
            
            # write csv log file
            if self.hps.csv_log:
                # construct an output string
                csv_outstr = "epoch,%d, step,%d, total,%.6E,%.6E, \
                recon,%.6E,%.6E,%.6E,%.6E, R^2 (Held-in, Held-out), %.6E, %.6E, %.6E, %.6E," \
                             "kl,%.6E,%.6E, l2,%.6E, klweight,%.6E, l2weight,%.6E\n"% \
                (nepoch, train_step, tr_total_cost, val_total_cost,
                 tr_recon_cost_train, tr_recon_cost_valid, trial_val_recon_cost_train, trial_val_recon_cost_valid,
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
            n_lr = hps['learning_rate_n_to_compare']
            train_cost_to_use = tr_total_cost
            if len(train_costs) > n_lr and train_cost_to_use > np.max(train_costs[-n_lr:]):
                print("Decreasing learning rate")
                self.run_learning_rate_decay_opt()

            train_costs.append(train_cost_to_use)

            nepoch += 1


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
          print("Setting number of posterior samples to process to : %d" % E)
          E_to_process = E

        # make a bunch of placeholders to store the posterior sample means
        if hps.ic_dim > 0:
            prior_g0_mean = [] #np.zeros([E_to_process, hps.ic_dim])
            prior_g0_logvar = [] #np.zeros([E_to_process, hps.ic_dim])
            post_g0_mean = [] #np.zeros([E_to_process, hps.ic_dim])
            post_g0_logvar = [] #np.zeros([E_to_process, hps.ic_dim])

        if hps.co_dim > 0:
            controller_outputs = [] #np.zeros([E_to_process, T, hps.co_dim])
        gen_ics = [] # np.zeros([E_to_process, hps.gen_dim])
        gen_states = [] #np.zeros([E_to_process, T, hps.gen_dim])
        factors = [] #np.zeros([E_to_process, T, hps.factors_dim])

        out_dist_params = [] #np.zeros([E_to_process, T, D])

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
            #print("Running %d of %d." % (es_idx+1, E_to_process))
            #example_idxs = es_idx * np.ones(batch_size, dtype=np.int32)
            # run the model
            mv = self.eval_model_runs_batch(data_name, data_bxtxd,
                                            do_eval_cost=True,
                                            do_average_batch=do_average_batch)
            # assemble a dict from the returns
            model_values = {}
            vars = ['gen_ics', 'gen_states', 'factors', 'output_dist_params']
            if self.hps.ic_dim > 0:
                vars.append('prior_g0_mean')
                vars.append('prior_g0_logvar')
                vars.append('post_g0_mean')
                vars.append('post_g0_logvar')
            if self.hps.co_dim > 0:
                vars.append('controller_outputs')

            for idx in range( len(vars) ):
                model_values[ vars[idx] ] = mv[idx] #, axis=0)


            #model_values['gen_ics'] = mv[idx]; idx += 1
            #model_values['gen_states'] = mv[idx]; idx += 1
            #model_values['factors'] = mv[idx]; idx += 1
            #model_values['output_dist_params'] = mv[idx]; idx += 1
            #if self.hps.ic_dim > 0:
            #    model_values['prior_g0_mean'] = mv[idx]; idx += 1
            #    model_values['prior_g0_logvar'] = mv[idx]; idx += 1
            #    model_values['post_g0_mean'] = mv[idx]; idx += 1
            #    model_values['post_g0_logvar'] = mv[idx]; idx += 1
            #if self.hps.co_dim > 0:
            #    model_values['controller_outputs'] = mv[idx]; idx += 1

            # assign values to the arrays
            if self.hps.ic_dim > 0:
                prior_g0_mean.append(model_values['prior_g0_mean'])
                prior_g0_logvar.append(model_values['prior_g0_logvar'])
                post_g0_mean.append(model_values['post_g0_mean'])
                post_g0_logvar.append(model_values['post_g0_logvar'])
            gen_ics.append(model_values['gen_ics'])

            if self.hps.co_dim > 0:
                controller_outputs.append(model_values['controller_outputs'])
            gen_states.append(model_values['gen_states'])
            factors.append(model_values['factors'])
            out_dist_params.append(model_values['output_dist_params'])

            ## these are from the original LFADS code, but not currently
            # computed in this version

            #costs[es_idx] = model_values['costs']
            #nll_bound_vaes[es_idx] = model_values['nll_bound_vaes']
            #nll_bound_iwaes[es_idx] = model_values['nll_bound_iwaes']
            #train_steps[es_idx] = model_values['train_steps']
            #print('bound nll(vae): %.3f, bound nll(iwae): %.3f' \
            #      % (nll_bound_vaes[es_idx], nll_bound_iwaes[es_idx]))
        print("")
        model_runs = {}
        if self.hps.ic_dim > 0:
            model_runs['prior_g0_mean'] = np.mean(prior_g0_mean, axis=0)
            model_runs['prior_g0_logvar'] = np.mean(prior_g0_logvar, axis=0)
            model_runs['post_g0_mean'] = np.mean(post_g0_mean, axis=0)
            model_runs['post_g0_logvar'] = np.mean(post_g0_logvar, axis=0)
            model_runs['gen_ics'] = np.mean(gen_ics, axis=0)

        if self.hps.co_dim > 0:
            model_runs['controller_outputs'] = np.mean(controller_outputs, axis=0)
        model_runs['gen_states'] = np.mean(gen_states, axis=0)
        model_runs['factors'] = np.mean(factors, axis=0)
        model_runs['output_dist_params'] = np.mean(out_dist_params, axis=0)
        #model_runs['costs'] = np.mean(costs, axis=0)
        #model_runs['nll_bound_vaes'] = np.mean(nll_bound_vaes, axis=0)
        #model_runs['nll_bound_iwaes'] = np.mean(nll_bound_iwaes, axis=0)
        #model_runs['train_steps'] = np.mean(train_steps, axis=0)
        return model_runs

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

        return R2_heldin, R2_heldout


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

            print("Writing data for %s data and kind %s to file %s." % (data_name, data_kind, fname))
            model_runs = self.eval_model_runs_avg_epoch(data_name, data_extxd)
            all_model_runs.append(model_runs)
            full_fname = os.path.join(hps.lfads_save_dir, fname)
            write_data(full_fname, model_runs, compression='gzip')
            print("Done.")

        return all_model_runs

            
