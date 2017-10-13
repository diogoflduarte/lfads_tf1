import tensorflow as tf
import numpy as np
import sys
import time


# utils defined by CP
from helper_funcs import linear, init_linear_transform, makeInitialState
from helper_funcs import ListOfRandomBatches, kind_dict
from helper_funcs import DiagonalGaussianFromInput, DiagonalGaussian, DiagonalGaussianFromExisting
from helper_funcs import BidirectionalDynamicRNN, DynamicRNN, LinearTimeVarying
from helper_funcs import KLCost_GaussianGaussian, Poisson
from plot_funcs import plot_data, close_all_plots
from data_funcs import read_datasets
from complexcell import ComplexCell





class LFADS(object):

    def __init__(self, hps, ):
        tf.reset_default_graph();

        self.learning_rate = tf.Variable(float(hps['learning_rate_init']), trainable=False, name="learning_rate")
        self.learning_rate_decay_op = self.learning_rate.assign(\
            self.learning_rate * hps['learning_rate_decay_factor'])


        ### BEGIN MODEL CONSTRUCTION

        allsets = hps['dataset_dims'].keys()
        self.input_dim = hps['dataset_dims'][allsets[0]]
        self.sequence_lengths = hps['sequence_lengths']

        with tf.variable_scope('placeholders'):
            self.input_data = tf.placeholder(tf.float32, shape = [hps['batch_size'], hps['num_steps'], self.input_dim], name='input_data')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.run_type = tf.placeholder(tf.int16, name='run_type')

        with tf.variable_scope('ic_enc'):
            ## ic_encoder
            self.ic_enc_rnn_obj = BidirectionalDynamicRNN(state_dim = hps['ic_enc_dim'],
                                                 inputs = self.input_data,
                                                 sequence_lengths = self.sequence_lengths,
                                                 batch_size = hps['batch_size'],
                                                 name = 'ic_enc',
                                                 rnn_type = 'gru',
                                                 output_keep_prob = self.keep_prob)

            # map the ic_encoder onto the actual ic layer
            self.ics_posterior = DiagonalGaussianFromInput(x = self.ic_enc_rnn_obj.last_tot,
                                        z_size = hps['ic_dim'],
                                        name = 'ic_enc_2_ics',
                                        var_min = hps['ic_var_min'],
                                        )

        # to go forward, either sample from the posterior, or take mean
        #     (depending on current usage)
        if self.run_type in [kind_dict("train"), kind_dict("posterior_sample_and_average")]:
            self.ics = self.ics_posterior.sample()
        else:
            self.ics = self.ics_posterior.mean
        with tf.variable_scope('generator'):
            self.g0 = linear(self.ics, hps['gen_dim'], name='ics_2_g0')


        ### CONTROLLER construction
        # this should only be done if a controller is requested
        # if not, skip all these graph elements like so:
        if hps['co_dim'] !=0:
            with tf.variable_scope('ci_enc'):
                ## ci_encoder
                self.ci_enc_rnn_obj = BidirectionalDynamicRNN(state_dim = hps['ci_enc_dim'],
                                                 inputs = self.input_data,
                                                 sequence_lengths = self.sequence_lengths,
                                                 batch_size = hps['batch_size'],
                                                 name = 'ci_enc',
                                                 rnn_type = 'gru',
                                                 output_keep_prob = self.keep_prob)
                # states from bidirec RNN come out as a tuple. concatenate those:
                self.ci_enc_rnn_states = tf.concat(self.ci_enc_rnn_obj.states,
                                                   axis=2, name='ci_enc_rnn_states')

                ## take a linear transform of the ci_enc output
                #    this is to lower the dimensionality of the ci_enc
                with tf.variable_scope('ci_enc_2_co_in'):
                    # one input from the encoder, another will come back from factors
                    self.ci_enc_object = LinearTimeVarying(inputs=self.ci_enc_rnn_states,
                                                           output_size = hps['con_ci_enc_in_dim'],
                                                           transform_name = 'ci_enc_2_co_in',
                                                           output_name = 'ci_enc_output_concat',
                                                           nonlinearity = None)
                    self.ci_enc_outputs = self.ci_enc_object.output

        ## the controller, controller outputs, generator, and factors are implemented
        #     in one RNN whose individual cell is "complex"
        #  this is required do to feedback pathway from factors->controller.
        #    impossible to dynamically unroll with separate RNNs.
        with tf.variable_scope('complexcell'):
            # the "complexcell" architecture requires an initial state definition
            # have to define states for each of the components, then concatenate them
            con_init_state = makeInitialState(hps['con_dim'],
                                              hps['batch_size'],
                                              'controller')
            co_mean_init_state = makeInitialState(hps['co_dim'],
                                                  hps['batch_size'],
                                                  'controller_output_mean')
            co_logvar_init_state = makeInitialState(hps['co_dim'],
                                                    hps['batch_size'],
                                                    'controller_output_logvar')
            fac_init_state = makeInitialState(hps['factors_dim'],
                                              hps['batch_size'],
                                              'factor')
            comcell_init_state = [con_init_state, self.g0,
                                       co_mean_init_state, co_logvar_init_state,
                                       fac_init_state]
            self.complexcell_init_state = tf.concat(axis=1, values = comcell_init_state)


            # here is what the state vector will look like
            self.comcell_state_dims = [hps['con_dim'],
                                       hps['gen_dim'],
                                       hps['co_dim'], # for the controller output means
                                       hps['co_dim'], # for the variances
                                       hps['factors_dim']]

            # construct the complexcell
            self.complexcell=ComplexCell(hps['con_dim'],
                                    hps['gen_dim'],
                                    hps['co_dim'],
                                    hps['factors_dim'],
                                    hps['con_fac_in_dim'],
                                    hps['batch_size'],
                                    var_min = hps['co_var_min'],
                                    kind = self.run_type)

            # construct the actual RNN
            #   its inputs are the output of the controller_input_enc
            self.complex_outputs, self.complex_final_state =\
            tf.nn.dynamic_rnn(self.complexcell,
                              inputs = self.ci_enc_outputs,
                              initial_state = self.complexcell_init_state,
                              time_major=False)

            self.con_states, self.gen_states, self.co_mean_states, self.co_logvar_states, self.fac_states =\
            tf.split(self.complex_outputs,
                     self.comcell_state_dims,
                     axis=2)

        with tf.variable_scope('rates'):
            ## rates
            rates_object = LinearTimeVarying(inputs = self.fac_states,
                                           output_size = self.input_dim,
                                           transform_name = 'factors_2_rates',
                                           output_name = 'rates_concat',
                                           nonlinearity = 'exp')
            # get both the pre-exponentiated and exponentiated versions
            self.logrates=rates_object.output
            self.rates=rates_object.output_nl



        ## calculate the KL cost
        # build a prior distribution to compare to
        self.ics_prior = DiagonalGaussian(z_size = [hps['batch_size'], hps['ic_dim']], name='ics_prior', var = hps['ic_var_min'])
        self.cos_prior = DiagonalGaussian(z_size = [hps['batch_size'], hps['num_steps'], hps['co_dim']], name='cos_prior', var = hps['co_var_min'])
        self.cos_posterior = DiagonalGaussianFromExisting(self.co_mean_states, \
                                                          self.co_logvar_states)

        # print ics_prior.mean_bxn
        # print ics_posterior.mean_bxn
        # print cos_prior.mean_bxn
        # print cos_posterior.mean_bxn
        # tmp = cos_posterior.mean_bxn - cos_prior.mean_bxn
        # print tmp
        # sys.exit()

        # cost for each trial
        self.kl_cost_g0_b = KLCost_GaussianGaussian(self.ics_posterior, self.ics_prior).kl_cost_b
        self.kl_cost_co_b = KLCost_GaussianGaussian(self.cos_posterior, self.cos_prior).kl_cost_b

        # total KL cost
        self.kl_cost_g0 = tf.reduce_mean(self.kl_cost_g0_b)
        self.kl_cost_co = tf.reduce_mean( tf.reduce_mean(self.kl_cost_co_b, [1]) )
        self.kl_cost = self.kl_cost_g0 + self.kl_cost_co

        ## calculate reconstruction cost
        self.loglikelihood_b_t = Poisson(self.logrates).logp(self.input_data)
        # cost for each trial
        self.log_p_b = tf.reduce_sum( tf.reduce_sum(self.loglikelihood_b_t, [2] ), [1] )
        # total rec cost
        self.log_p = tf.reduce_mean( self.log_p_b, [0])
        self.rec_cost = -self.log_p

        ## calculate total cost
        self.total_cost = self.kl_cost + self.rec_cost

        # get the list of trainable variables
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients = tf.gradients(self.total_cost, self.trainable_vars)
        self.gradients, self.grad_global_norm = tf.clip_by_global_norm(self.gradients, \
                                                                       hps['max_grad_norm'])
        self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999,
                                     epsilon=1e-01)

        self.train_step = tf.get_variable("global_step", [], tf.int64,
                                     tf.zeros_initializer(),
                                     trainable=False)
        self.train_op = self.opt.apply_gradients(
            zip(self.gradients, self.trainable_vars), global_step = self.train_step)



## functions to interface with the outside world
        
    def train_batch(self, dict_from_py):
        session = tf.get_default_session()
        ops_to_eval = [self.train_op, self.total_cost, self.rec_cost, \
                             self.kl_cost, self.rates, self.learning_rate]
        feed_dict = {self.input_data: dict_from_py['input_data'],
                     self.keep_prob: dict_from_py['keep_prob'],
                     self.run_type: kind_dict("train")}
        return session.run(ops_to_eval, feed_dict)


    def validation_batch(self, dict_from_py):
        session = tf.get_default_session()
        ops_to_eval = [self.total_cost, self.rec_cost, self.kl_cost]
        feed_dict = {self.input_data: dict_from_py['input_data'],
                     self.keep_prob: 1.0, # don't need to lower keep_prob from validation
                     self.run_type: kind_dict("train")}
        return session.run(ops_to_eval, feed_dict)

    def get_controller_output(self, dict_from_py):
        session = tf.get_default_session()
        ops_to_eval = [self.co_mean_states]
        feed_dict = {self.input_data: dict_from_py['input_data'],
                     self.keep_prob: 1.0, # don't need to lower keep_prob for this
                     self.run_type: kind_dict("train")}
        return session.run(ops_to_eval, feed_dict)
    

    def run_learning_rate_decay_opt(self):
        session = tf.get_default_session()
        session.run(self.learning_rate_decay_op)

    def get_learning_rate(self):
        session = tf.get_default_session()
        return session.run(self.learning_rate)
