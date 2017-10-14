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




class SimpleModel(object):
    def __init__(self, hps, ):
        tf.reset_default_graph();

        self.learning_rate = tf.Variable(float(hps['learning_rate_init']), trainable=False, name="learning_rate")
        self.learning_rate_decay_op = self.learning_rate.assign(\
            self.learning_rate * hps['learning_rate_decay_factor'])

        allsets = hps['dataset_dims'].keys()
        self.input_dim = hps['dataset_dims'][allsets[0]]
        self.sequence_lengths = hps['sequence_lengths']

        with tf.variable_scope('placeholders'):
            self.input_data = tf.placeholder(tf.float32, shape = [hps['batch_size'], \
                                                                  hps['num_steps'], \
                                                                  self.input_dim], \
                                             name='input_data')

            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.run_type = tf.placeholder(tf.int16, name='run_type')


        with tf.variable_scope('ic_enc'):
            ## ic_encoder
            self.ic_enc_rnn_obj = BidirectionalDynamicRNN(
                state_dim = hps['ic_enc_dim'],
                sequence_lengths = self.sequence_lengths,
                batch_size = hps['batch_size'],
                name = 'ic_enc',
                inputs = self.input_data,
                initial_state = None,
                rnn_type = 'gru',
                output_keep_prob = self.keep_prob)

            
            # map the ic_encoder onto the actual ic layer
            self.ics_posterior = DiagonalGaussianFromInput(x = self.ic_enc_rnn_obj.last_tot,
                                        z_size = hps['ic_dim'],
                                        name = 'ic_enc_2_ics',
                                        var_min = hps['ic_var_min'],
                                        )
            self.ics_prior = DiagonalGaussian(z_size = [hps['batch_size'], hps['ic_dim']], \
                                              name='ics_prior', var = hps['ic_var_min'])


        # to go forward, either sample from the posterior, or take mean
        #     (depending on current usage)
        if self.run_type in [kind_dict("train"), kind_dict("posterior_sample_and_average")]:
            self.ics = self.ics_posterior.sample()
        else:
            self.ics = self.ics_posterior.mean




        with tf.variable_scope('generator'):
            self.g0 = linear(self.ics, hps['gen_dim'], name='ics_2_g0')

            # setup the actual generator
            self.gen_rnn_obj = DynamicRNN(state_dim = hps['gen_dim'],
                                          sequence_lengths = self.sequence_lengths,
                                          batch_size = hps['batch_size'],
                                          name = 'gen',
                                          initial_state = self.g0,
                                          inputs = None,
                                          rnn_type = 'gru',
                                          output_keep_prob = self.keep_prob)

            ## factors
            self.factors = LinearTimeVarying(inputs = self.gen_rnn_obj.states,
                                                 output_size = hps['factors_dim'],
                                                 transform_name = 'gen_2_factors',
                                                 output_name = 'factors_concat',
                                                 )


            with tf.variable_scope('rates'):
                ## rates
                rates_object = LinearTimeVarying(inputs = self.factors.output,
                                                 output_size = self.input_dim,
                                                 transform_name = 'factors_2_rates',
                                                 output_name = 'rates_concat',
                                                 nonlinearity = 'exp')
                
            # get both the pre-exponentiated and exponentiated versions
            self.logrates = rates_object.output
            self.rates = rates_object.output_nl

            
        ## calculate reconstruction cost
        self.loglikelihood_b_t_n = Poisson(self.logrates).logp(self.input_data)
        # cost for each trial
        self.log_p_b = tf.reduce_sum( tf.reduce_sum(self.loglikelihood_b_t_n, [2] ), [1] )
            
        # total rec cost
        self.log_p = tf.reduce_mean( self.log_p_b, [0]) # log likelihood (higher is better)
        self.rec_cost = -self.log_p # negative log likelihood (lower is better)

            
        # kl cost for each trial
        self.kl_cost_g0_b = KLCost_GaussianGaussian(self.ics_posterior, self.ics_prior).kl_cost_b
        # total kl cost 
        self.kl_cost = tf.reduce_mean(self.kl_cost_g0_b) # kl cost (lower is better)

        
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



    def run_learning_rate_decay_opt(self):
        session = tf.get_default_session()
        session.run(self.learning_rate_decay_op)

    def get_learning_rate(self):
        session = tf.get_default_session()
        return session.run(self.learning_rate)
    
