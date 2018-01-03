import tensorflow as tf
import numpy as np
import sys
import time
import os
import random


# utils defined by CP
from helper_funcs import linear, init_linear_transform, makeInitialState
from helper_funcs import ListOfRandomBatches, kind_dict, kind_dict_key
from helper_funcs import DiagonalGaussianFromInput, DiagonalGaussian, DiagonalGaussianFromExisting, diag_gaussian_log_likelihood
from helper_funcs import BidirectionalDynamicRNN, DynamicRNN, LinearTimeVarying
from helper_funcs import KLCost_GaussianGaussian, Poisson
from helper_funcs import write_data
from plot_funcs import plot_data, close_all_plots
from data_funcs import read_datasets
from customcells import ComplexCell
from customcells import CustomGRUCell






class LFADS(object):

    def __init__(self, hps, datasets = None):
    # build the graph
        
        # set the learning rate, defaults to the hyperparam setting
        # TODO: test if this properly re-initializes learning rate when loading a new model
        self.learning_rate = tf.Variable(float(hps['learning_rate_init']), trainable=False, name="learning_rate")

        # this is how the learning rate is decayed over time
        self.learning_rate_decay_op = self.learning_rate.assign(\
            self.learning_rate * hps['learning_rate_decay_factor'])


        ### BEGIN MODEL CONSTRUCTION

        # figure out the input (dataset) dimensionality
        # TODO: adjust so this can handle multiple datasets
        allsets = hps['dataset_dims'].keys()
        self.input_dim = hps['dataset_dims'][allsets[0]]

        # everybody uses this variable to get the batch size
        # in the future, this could be dynamic 
        graph_batch_size = hps['batch_size']

        # define all placeholders
        with tf.variable_scope('placeholders'):
            # input data (what are we training on)
            self.dataset_ph = tf.placeholder(tf.float32, shape = [graph_batch_size, hps['num_steps'], self.input_dim], name='input_data')
            # dropout keep probability
            #   enumerated in helper_funcs.kind_dict
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            
            self.run_type = tf.placeholder(tf.int16, name='run_type')
            self.kl_ic_weight = tf.placeholder(tf.float32, name='kl_ic_weight')
            self.kl_co_weight = tf.placeholder(tf.float32, name='kl_co_weight')
            # name of the dataset
            self.dataName = tf.placeholder(tf.string, shape=(), name='dataset_name')

        with tf.variable_scope('ic_enc'):
            ic_enc_cell = CustomGRUCell(num_units = hps['ic_enc_dim'],\
                                        batch_size = graph_batch_size,
                                        clip_value = hps['cell_clip_value'])
            ## ic_encoder
            self.ic_enc_rnn_obj = BidirectionalDynamicRNN(
                state_dim = hps['ic_enc_dim'],
                batch_size = graph_batch_size,
                name = 'ic_enc',
                sequence_lengths = np.zeros(graph_batch_size) + hps['num_steps'],
                cell = ic_enc_cell,
                inputs = self.dataset_ph,
                initial_state = None,
                rnn_type = 'customgru',
                output_keep_prob = self.keep_prob)

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
                gen_cell = CustomGRUCell(num_units = hps['ic_enc_dim'],\
                                            batch_size = graph_batch_size,
                                            clip_value = hps['cell_clip_value'])
                # setup generator
                self.gen_rnn_obj = DynamicRNN(state_dim = hps['gen_dim'],
                                              batch_size = graph_batch_size,
                                              name = 'gen',
                                              sequence_lengths = np.zeros(graph_batch_size) + hps['num_steps'],
                                              cell = gen_cell,
                                              inputs = None,
                                              initial_state = self.gen_ics,
                                              rnn_type = 'customgru',
                                              output_keep_prob = self.keep_prob)
            with tf.variable_scope('factors'):

                ## factors
                self.fac_obj = LinearTimeVarying(inputs = self.gen_rnn_obj.states,
                                                    output_size = hps['factors_dim'],
                                                    transform_name = 'gen_2_factors',
                                                    output_name = 'factors_concat',
                                                 )
                self.factors = self.fac_obj.output
        else:
            with tf.variable_scope('ci_enc'):

                ci_enc_cell = CustomGRUCell(num_units = hps['ci_enc_dim'],\
                                            batch_size = graph_batch_size,
                                            clip_value = hps['cell_clip_value'])
                ## ci_encoder
                self.ci_enc_rnn_obj = BidirectionalDynamicRNN(
                    state_dim = hps['ci_enc_dim'],
                    batch_size = graph_batch_size,
                    name = 'ci_enc',
                    sequence_lengths = np.zeros(graph_batch_size) + hps['num_steps'],
                    cell = ci_enc_cell,
                    inputs = self.dataset_ph,
                    initial_state = None,
                    rnn_type = 'customgru',
                    output_keep_prob = self.keep_prob)

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
                output_size = self.input_dim
            elif hps.output_dist.lower() == 'gaussian':
                nonlin = None
                output_size = self.input_dim * 2
            else:
                raise NameError("Unknown output distribution: " + hps.output_dist)
                
            # rates are taken as a linear (or nonlinear) readout from the factors
            rates_object = LinearTimeVarying(inputs = self.factors,
                                             output_size = output_size,
                                             transform_name = 'factors_2_rates',
                                             output_name = 'rates_concat',
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

        ## calculate reconstruction cost
        if hps.output_dist.lower() == 'poisson':
            self.loglikelihood_b_t = Poisson(self.logrates).logp(self.dataset_ph)
        if hps.output_dist.lower() == 'gaussian':
            self.loglikelihood_b_t = diag_gaussian_log_likelihood(self.dataset_ph,
                                         self.output_mean, self.output_logvar)
            
        # cost for each trial
        self.log_p_b = tf.reduce_sum(
            tf.reduce_sum(self.loglikelihood_b_t, [2] ), [1] )
        # total rec cost (avg over all trials in batch)
        self.log_p = tf.reduce_mean( self.log_p_b, [0])
        self.rec_cost = -self.log_p

        ## calculate total cost
        self.total_cost = self.kl_cost + self.rec_cost

        # get the list of trainable variables
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
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
                        keep_prob=None, kl_ic_weight=1.0, kl_co_weight=1.0):
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
        ops_to_eval = [self.total_cost, self.rec_cost,
                       self.kl_cost]
        # get a full list of all data for this type (train/valid)
        all_name_example_idx_pairs = \
          self.shuffle_and_flatten_datasets(datasets, train_or_valid)
        
        if train_or_valid == "train":
            ops_to_eval.append(self.train_op)
            kind_data = "train_data"
            keep_prob = self.hps.keep_prob
        else:
            kind_data = "valid_data"
            keep_prob = 1.0
            
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
                                             kl_co_weight = kl_co_weight)
            evald_ops_this_batch = session.run(ops_to_eval, feed_dict = feed_dict)
            # for training runs, there is an extra output argument. kill it
            if len(evald_ops_this_batch) > 3:
                tc, rc, kl, _= evald_ops_this_batch
                evald_ops_this_batch = (tc, rc, kl)
            evald_ops.append(evald_ops_this_batch)
        evald_ops = np.mean(evald_ops, axis=0)
        return evald_ops
        
        
    def train_batch(self, dict_from_py):
        session = tf.get_default_session()
        ops_to_eval = [self.train_op, self.total_cost, self.rec_cost, \
                             self.kl_cost, self.output_dist_params, self.learning_rate]
        feed_dict = {self.dataset_ph: dict_from_py['dataset_ph'],
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
    # decay the learning rate 
        session = tf.get_default_session()
        session.run(self.learning_rate_decay_op)

    def get_learning_rate(self):
    # return the current learning rate
        session = tf.get_default_session()
        return session.run(self.learning_rate)


    def train_model(self, datasets, num_epochs=None):
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
        nepoch = -1

        # TODO: modify to work with multiple datasets
        name=datasets.keys()[0]
        
        data_dict = datasets[name]
        data_extxd = data_dict["train_data"]
        this_batch = data_extxd[0:hps['batch_size'],:,:]

        train_costs = []
        lve = float('Inf')
        while True:
            nepoch += 1
            do_save_ckpt = True if nepoch % 10 == 0 else False
            start_time = time.time()
            tr_total_cost, tr_recon_cost, tr_kl_cost = \
                self.train_epoch(datasets, do_save_ckpt=do_save_ckpt,
                                 kl_ic_weight = hps['kl_ic_weight'],
                                 kl_co_weight = hps['kl_co_weight'])
            epoch_time = time.time() - start_time
            print("Elapsed time: %f" %epoch_time)

            val_total_cost, val_recon_cost, val_kl_cost = \
                self.valid_epoch(datasets,
                                 kl_ic_weight = hps['kl_ic_weight'],
                                 kl_co_weight = hps['kl_co_weight'])
            if val_recon_cost < lve:
                # new lowest validation error
                lve = val_recon_cost
                # MRK, make lve accessible from the model class
                self.lve = lve

                checkpoint_path = os.path.join(self.hps.lfads_save_dir,
                                               self.hps.checkpoint_name + '_lve.ckpt')
                # MRK, for convenience, it can be reconstructed from the paths in hps
                #self.lve_checkpoint = checkpoint_path

                self.lve_saver.save(session, checkpoint_path,
                                    global_step=self.train_step,
                                    latest_filename='checkpoint_lve')
            
            kl_weight = hps['kl_ic_weight']
            train_step = session.run(self.train_step)
            print("Epoch:%d, step:%d (TRAIN, VALID): total: %.2f, %.2f\
            recon: %.2f, %.2f,     kl: %.2f, %.2f, kl weight: %.2f" % \
                  (nepoch, train_step, tr_total_cost, val_total_cost,
                   tr_recon_cost, val_recon_cost, tr_kl_cost, val_kl_cost,
                   kl_weight))

            #l2 has not been implemented yet
            l2_cost = 0
            l2_weight = 0

            # TODO: write tensorboard summaries here...
            
            # write csv log file
            if self.hps.csv_log:
                # construct an output string
                csv_outstr = "epoch,%d, step,%d, total,%.2f,%.2f, \
                recon,%.2f,%.2f, kl,%.2f,%.2f, l2,%.5f, \
                klweight,%.2f, l2weight,%.2f\n"% \
                (nepoch, train_step, tr_total_cost, val_total_cost,
                 tr_recon_cost, val_recon_cost, tr_kl_cost, val_kl_cost,
                 l2_cost, kl_weight, l2_weight)
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
                
            new_lr = self.get_learning_rate()
            # should we stop?
            if num_epochs is None:
                if new_lr < hps['learning_rate_stop']:
                    print("Learning rate criteria met")
                    break
            else:
                if nepoch == num_epochs + 1:  # nepoch starts at 0
                    print("Num epoch criteria met. "
                          "Completed {} epochs.".format(nepoch))
                    break



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
        feed_dict = self.build_feed_dict(data_name, data_bxtxd, run_type=kind_dict('posterior_sample_and_average'),
                                         keep_prob=1.0) 
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
    def eval_model_runs_avg_epoch(self, data_name, data_extxd):
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
        E, T, D  = data_extxd.shape
        E_to_process = hps.ps_nexamples_to_process
        if E_to_process > E:
          print("Setting number of posterior samples to process to : ", E)
          E_to_process = E

        # make a bunch of placeholders to store the posterior sample means
        if hps.ic_dim > 0:
            prior_g0_mean = np.zeros([E_to_process, hps.ic_dim])
            prior_g0_logvar = np.zeros([E_to_process, hps.ic_dim])
            post_g0_mean = np.zeros([E_to_process, hps.ic_dim])
            post_g0_logvar = np.zeros([E_to_process, hps.ic_dim])

        if hps.co_dim > 0:
            controller_outputs = np.zeros([E_to_process, T, hps.co_dim])
        gen_ics = np.zeros([E_to_process, hps.gen_dim])
        gen_states = np.zeros([E_to_process, T, hps.gen_dim])
        factors = np.zeros([E_to_process, T, hps.factors_dim])

        if hps.output_dist == 'poisson':
            out_dist_params = np.zeros([E_to_process, T, D])
        elif hps.output_dist == 'gaussian':
            out_dist_params = np.zeros([E_to_process, T, D+D])
        else:
            assert False, "NIY"

        costs = np.zeros(E_to_process)
        nll_bound_vaes = np.zeros(E_to_process)
        nll_bound_iwaes = np.zeros(E_to_process)
        train_steps = np.zeros(E_to_process)
        for es_idx in range(E_to_process):
            print("Running %d of %d." % (es_idx+1, E_to_process))
            example_idxs = es_idx * np.ones(batch_size, dtype=np.int32)
            data_bxtxd = data_extxd[example_idxs]
            # run the model
            mv = self.eval_model_runs_batch(data_name, data_bxtxd,
                                            do_eval_cost=True,
                                            do_average_batch=True)
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
                model_values[ vars[idx] ] = np.mean(mv[idx], axis=0)

                
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
                prior_g0_mean[es_idx,:] = model_values['prior_g0_mean']
                prior_g0_logvar[es_idx,:] = model_values['prior_g0_logvar']
                post_g0_mean[es_idx,:] = model_values['post_g0_mean']
                post_g0_logvar[es_idx,:] = model_values['post_g0_logvar']
            gen_ics[es_idx,:] = model_values['gen_ics']

            if self.hps.co_dim > 0:
                controller_outputs[es_idx,:,:] = model_values['controller_outputs']
            gen_states[es_idx,:,:] = model_values['gen_states']
            factors[es_idx,:,:] = model_values['factors']
            out_dist_params[es_idx,:,:] = model_values['output_dist_params']

            ## these are from the original LFADS code, but not currently
            # computed in this version
            
            #costs[es_idx] = model_values['costs']
            #nll_bound_vaes[es_idx] = model_values['nll_bound_vaes']
            #nll_bound_iwaes[es_idx] = model_values['nll_bound_iwaes']
            #train_steps[es_idx] = model_values['train_steps']
            #print('bound nll(vae): %.3f, bound nll(iwae): %.3f' \
            #      % (nll_bound_vaes[es_idx], nll_bound_iwaes[es_idx]))

        model_runs = {}
        if self.hps.ic_dim > 0:
            model_runs['prior_g0_mean'] = prior_g0_mean
            model_runs['prior_g0_logvar'] = prior_g0_logvar
            model_runs['post_g0_mean'] = post_g0_mean
            model_runs['post_g0_logvar'] = post_g0_logvar
            model_runs['gen_ics'] = gen_ics

        if self.hps.co_dim > 0:
            model_runs['controller_outputs'] = controller_outputs
        model_runs['gen_states'] = gen_states
        model_runs['factors'] = factors
        model_runs['output_dist_params'] = out_dist_params
        model_runs['costs'] = costs
        model_runs['nll_bound_vaes'] = nll_bound_vaes
        model_runs['nll_bound_iwaes'] = nll_bound_iwaes
        model_runs['train_steps'] = train_steps
        return model_runs



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

        for data_name, data_dict in datasets.items():
          data_tuple = [('train', data_dict['train_data']),
                        ('valid', data_dict['valid_data'])]
          for data_kind, data_extxd in data_tuple:
            if not output_fname:
              fname = "model_runs_" + data_name + '_' + data_kind + '_' + kind
            else:
              fname = output_fname + data_name + '_' + data_kind + '_' + kind

            print("Writing data for %s data and kind %s." % (data_name, data_kind))
            model_runs = self.eval_model_runs_avg_epoch(data_name, data_extxd)
            full_fname = os.path.join(hps.lfads_save_dir, fname)
            write_data(full_fname, model_runs, compression='gzip')
            print("Done.")

            
