from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import h5py
import json
import sys
import warnings
import errno


def kind_dict_definition():
    # used in the graph's keep probability
    return {
        'train': 1,
        'posterior_sample_and_average': 2,
        'posterior_mean': 3,
        'prior_sample': 4,
        'write_model_params': 5,
    }


def kind_dict(kind_str):
    # used in the graph's keep probability
    kd = kind_dict_definition()
    return kd[kind_str]


def kind_dict_key(kind_number):
    # get the key for a certain ind
    kd = kind_dict_definition()
    for key, val in kd.iteritems():
        if val == kind_number:
            return key


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def printer(data):
    # prints on the same line
    sys.stdout.write("\r\x1b[K" + data.__str__())
    sys.stdout.flush()


def write_data(data_fname, data_dict, use_json=False, compression=None):
    """Write data in HDF5 format.

    Args:
      data_fname: The filename of teh file in which to write the data.
      data_dict:  The dictionary of data to write. The keys are strings
        and the values are numpy arrays.
      use_json (optional): human readable format for simple items
      compression (optional): The compression to use for h5py (disabled by
        default because the library borks on scalars, otherwise try 'gzip').
    """

    dir_name = os.path.dirname(data_fname)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if use_json:
        the_file = open(data_fname, 'w')
        json.dump(data_dict, the_file)
        the_file.close()
    else:
        try:
            with h5py.File(data_fname, 'w') as hf:
                for k, v in data_dict.items():
                    clean_k = k.replace('/', '_')
                    if clean_k is not k:
                        print('Warning: saving variable with name: ', k, ' as ', clean_k)
                    else:
                        print('Saving variable with name: ', clean_k)
                    hf.create_dataset(clean_k, data=v, compression=compression)
        except IOError:
            print("Cannot open %s for writing.", data_fname)
            raise


def init_linear_transform(in_size, out_size, name=None, collections=None, mat_init_value=None, bias_init_value=None):
    # generic function (we use linear transforms in a lot of places)
    # initialize the weights of the linear transformation based on the size of the inputs

    # initialze with a random distribuion
    stddev = 1 / np.sqrt(float(in_size))
    mat_init = tf.random_normal_initializer(0.0, stddev, dtype=tf.float32)

    # weight matrix
    w_collections = [tf.GraphKeys.GLOBAL_VARIABLES, "norm-variables"]
    if collections:
        w_collections += collections
    wname = (name + "/W") if name else "/W"
    w = tf.get_variable(wname, [in_size, out_size], initializer=mat_init,
                        dtype=tf.float32, collections=w_collections)

    # biases
    bname = (name + "/b") if name else "/b"
    b = tf.get_variable(bname, [1, out_size],
                        initializer=tf.zeros_initializer(),
                        dtype=tf.float32)
    return (w, b)


# needed for LFADS's linear
def init_linear(in_size, out_size, do_bias=True, mat_init_value=None, alpha=1.0,
                identity_if_possible=False, normalized=False,
                name=None, collections=None):
    """Linear (affine) transformation, y = x W + b, for a variety of
    configurations.

    Args:
      in_size: The integer size of the non-batc input dimension. [(x),y]
      out_size: The integer size of non-batch output dimension. [x,(y)]
      do_bias (optional): Add a learnable bias vector to the operation.
      mat_init_value (optional): numpy constant for matrix initialization, if None
        , do random, with additional parameters.
      alpha (optional): A multiplicative scaling for the weight initialization
        of the matrix, in the form \alpha * 1/\sqrt{x.shape[1]}.
      identity_if_possible (optional): just return identity,
        if x.shape[1] == out_size.
      normalized (optional): Option to divide out by the norms of the rows of W.
      name (optional): The name prefix to add to variables.
      collections (optional): List of additional collections. (Placed in
        tf.GraphKeys.GLOBAL_VARIABLES already, so no need for that.)

    Returns:
      In the equation, y = x W + b, returns the pair (W, b).
    """

    if mat_init_value is not None and mat_init_value.shape != (in_size, out_size):
        raise ValueError(
            'Provided mat_init_value must have shape [%d, %d].' % (in_size, out_size))

    if mat_init_value is None:
        stddev = alpha / np.sqrt(float(in_size))
        mat_init = tf.random_normal_initializer(0.0, stddev)

    wname = (name + "/W") if name else "/W"

    if identity_if_possible and in_size == out_size:
        return (tf.constant(np.eye(in_size).astype(np.float32)),
                tf.zeros(in_size))

    # Note the use of get_variable vs. tf.Variable.  this is because get_variable
    # does not allow the initialization of the variable with a value.
    if normalized:
        w_collections = [tf.GraphKeys.GLOBAL_VARIABLES, "norm-variables"]
        if collections:
            w_collections += collections
        if mat_init_value is not None:
            w = tf.Variable(mat_init_value, name=wname, collections=w_collections)
        else:
            w = tf.get_variable(wname, [in_size, out_size], initializer=mat_init,
                                collections=w_collections)
        w = tf.nn.l2_normalize(w, dim=0)  # x W, so xW_j = \sum_i x_bi W_ij
    else:
        w_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if collections:
            w_collections += collections
        if mat_init_value is not None:
            w = tf.Variable(mat_init_value, name=wname, collections=w_collections)
        else:
            w = tf.get_variable(wname, [in_size, out_size], initializer=mat_init,
                                collections=w_collections)
    if do_bias:
        b_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if collections:
            b_collections += collections
        bname = (name + "/b") if name else "/b"
        b = tf.get_variable(bname, [1, out_size],
                            initializer=tf.zeros_initializer(),
                            )
        # don't add biases to collections for l2 penalty
        # collections=b_collections)
    else:
        b = None

    return (w, b)


# LFADS's version of linear
def linear2(x, out_size, do_bias=True, alpha=1.0, identity_if_possible=False,
            normalized=False, name=None, collections=None):
    """Linear (affine) transformation, y = x W + b, for a variety of
    configurations.

    Args:
      x: input The tensor to tranformation.
      out_size: The integer size of non-batch output dimension.
      do_bias (optional): Add a learnable bias vector to the operation.
      alpha (optional): A multiplicative scaling for the weight initialization
        of the matrix, in the form \alpha * 1/\sqrt{x.shape[1]}.
      identity_if_possible (optional): just return identity,
        if x.shape[1] == out_size.
      normalized (optional): Option to divide out by the norms of the rows of W.
      name (optional): The name prefix to add to variables.
      collections (optional): List of additional collections. (Placed in
        tf.GraphKeys.GLOBAL_VARIABLES already, so no need for that.)

    Returns:
      In the equation, y = x W + b, returns the tensorflow op that yields y.
    """
    in_size = int(x.get_shape()[1])  # from Dimension(10) -> 10
    stddev = alpha / np.sqrt(float(in_size))
    mat_init = tf.random_normal_initializer(0.0, stddev)
    wname = (name + "/W") if name else "/W"

    if identity_if_possible and in_size == out_size:
        # Sometimes linear layers are nothing more than size adapters.
        return tf.identity(x, name=(wname + '_ident'))

    W, b = init_linear(in_size, out_size, do_bias=do_bias, alpha=alpha,
                       normalized=normalized, name=name, collections=collections)

    if do_bias:
        return tf.matmul(x, W) + b
    else:
        return tf.matmul(x, W)


def linear(x, out_size, name, collections=None, mat_init_value=None, bias_init_value=None):
    # generic function (we use linear transforms in a lot of places)
    # initialize the weights of the linear transformation based on the size of the inputs
    in_size = int(x.get_shape()[1])
    W, b = init_linear_transform(in_size, out_size, name=name, collections=collections, mat_init_value=mat_init_value,
                                 bias_init_value=bias_init_value)
    return tf.matmul(x, W) + b


def ListOfRandomBatches(num_trials, batch_size):
    if num_trials <= batch_size:
        warnings.warn("Your batch size is bigger than num_trials! Using single batch ...")
        return [np.random.permutation(range(num_trials))]

    random_order = np.random.permutation(range(num_trials))
    even_num_of_batches = int(np.floor(num_trials / batch_size))
    trials_to_keep = even_num_of_batches * batch_size
    # if num_trials % batch_size != 0:
    #    print("Warning: throwing out %i trials per epoch" % (num_trials-trials_to_keep) )
    random_order = random_order[0:(trials_to_keep)]
    batches = [random_order[i:i + batch_size] for i in range(0, len(random_order), batch_size)]
    return batches


class Gaussian(object):
    """Base class for Gaussian distribution classes."""

    @property
    def mean(self):
        return self.mean_bxn

    @property
    def logvar(self):
        return self.logvar_bxn

    def noise(self):
        return self.noise_bxn

    def sample(self):
        return self.mean + tf.exp(0.5 * self.logvar) * self.noise()


def diag_gaussian_log_likelihood(z, mu=0.0, logvar=0.0):
    """Log-likelihood under a Gaussian distribution with diagonal covariance.
      Returns the log-likelihood for each dimension.  One should sum the
      results for the log-likelihood under the full multidimensional model.

    Args:
      z: The value to compute the log-likelihood.
      mu: The mean of the Gaussian
      logvar: The log variance of the Gaussian.

    Returns:
      The log-likelihood under the Gaussian model.
    """

    return -0.5 * (logvar + np.log(2 * np.pi) + \
                   tf.square((z - mu) / tf.exp(0.5 * logvar)))


class DiagonalGaussianFromExisting(Gaussian):
    """Diagonal Gaussian with different constant mean and variances in each
    dimension.
    """

    def __init__(self, mean_bxn, logvar_bxn):
        self.mean_bxn = mean_bxn
        self.logvar_bxn = logvar_bxn
        self.noise_bxn = tf.random_normal(tf.shape(self.logvar_bxn))


class DiagonalGaussian(Gaussian):
    """Diagonal Gaussian with different constant mean and variances in each
    dimension.
    """

    def __init__(self, z_size, name, var):
        self.mean_bxn = tf.zeros(z_size)
        self.logvar_bxn = tf.log(tf.zeros(z_size) + var)
        self.noise_bxn = tf.random_normal(tf.shape(self.logvar_bxn))


class DiagonalGaussianFromInput(Gaussian):
    """Diagonal Gaussian taken as a linear transform of input.
    """

    def __init__(self, x, z_size, name, var_min=0.0):
        # size_bxn = tf.stack([tf.shape(x)[0], z_size])
        self.mean_bxn = linear(x, z_size, name=(name + "/mean"))
        logvar_bxn = linear(x, z_size, name=(name + "/logvar"))
        if var_min > 0.0:
            logvar_bxn = tf.log(tf.exp(logvar_bxn) + var_min)
        self.logvar_bxn = logvar_bxn
        self.noise_bxn = tf.random_normal(tf.shape(self.logvar_bxn))


def makeInitialState(state_dim, batch_size, name):
    init_stddev = 1 / np.sqrt(float(state_dim))
    init_initter = tf.random_normal_initializer(0.0, init_stddev, dtype=tf.float32)
    init_state = tf.get_variable(name + '_init_state', [1, state_dim],
                                 initializer=init_initter,
                                 dtype=tf.float32, trainable=True)
    tile_dimensions = [batch_size, 1]
    init_state_tiled = tf.tile(init_state,
                               tile_dimensions,
                               name=name + '_init_state_tiled')
    return init_state_tiled


class BidirectionalDynamicRNN(object):
    def __init__(self, state_dim, batch_size, name, sequence_lengths,
                 cell=None, inputs=None, initial_state=None, rnn_type='gru',
                 output_size=None, output_keep_prob=1.0, input_keep_prob=1.0):

        if initial_state is None:
            # need initial states for fw and bw
            self.init_stddev = 1 / np.sqrt(float(state_dim))
            self.init_initter = tf.random_normal_initializer(0.0, self.init_stddev, dtype=tf.float32)

            self.init_h_fw = tf.get_variable(name + '_init_h_fw', [1, state_dim],
                                             initializer=self.init_initter,
                                             dtype=tf.float32)
            self.init_h_bw = tf.get_variable(name + '_init_h_bw', [1, state_dim],
                                             initializer=self.init_initter,
                                             dtype=tf.float32)
            # lstm has a second parameter c
            if rnn_type.lower() == 'lstm':
                self.init_c_fw = tf.get_variable(name + '_init_c_fw', [1, state_dim],
                                                 initializer=self.init_initter,
                                                 dtype=tf.float32)
                self.init_c_bw = tf.get_variable(name + '_init_c_bw', [1, state_dim],
                                                 initializer=self.init_initter,
                                                 dtype=tf.float32)

            tile_dimensions = [batch_size, 1]

            # tile the h param
            self.init_h_fw_tiled = tf.tile(self.init_h_fw,
                                           tile_dimensions, name=name + '_h_fw_tile')
            self.init_h_bw_tiled = tf.tile(self.init_h_bw,
                                           tile_dimensions, name=name + '_h_bw_tile')
            # tile the c param if needed
            if rnn_type.lower() == 'lstm':
                self.init_c_fw_tiled = tf.tile(self.init_c_fw,
                                               tile_dimensions, name=name + '_c_fw_tile')
                self.init_c_bw_tiled = tf.tile(self.init_c_bw,
                                               tile_dimensions, name=name + '_c_bw_tile')

            # do tupling if needed
            if rnn_type.lower() == 'lstm':
                # lstm state is a tuple
                self.init_fw = tf.contrib.rnn.LSTMStateTuple(self.init_c_fw_tiled, self.init_h_fw_tiled)
                self.init_bw = tf.contrib.rnn.LSTMStateTuple(self.init_c_bw_tiled, self.init_h_bw_tiled)
            else:
                self.init_fw = self.init_h_fw_tiled
                self.init_bw = self.init_h_bw_tiled
        else:  # if initial state is None
            self.init_fw, self.init_bw = initial_state

        # pick your cell
        if rnn_type.lower() == 'lstm':
            self.cell = tf.nn.rnn_cell.LSTMCell(num_units=state_dim,
                                                state_is_tuple=True)
        elif rnn_type.lower() == 'gru':
            self.cell = tf.nn.rnn_cell.GRUCell(num_units=state_dim)
        else:
            self.cell = cell

        # add dropout if requested
        if output_keep_prob != 1.0:
            self.cell = tf.contrib.rnn.DropoutWrapper(
                self.cell, output_keep_prob=output_keep_prob)

        # unused
        # convenient way to take a low-D output from network
        if output_size is not None:
            self.cell = tf.contrib.rnn.OutputProjectionWrapper(self.cell, output_size=output_size)

        # for some reason I can't get dynamic_rnn to work without inputs
        #  so generate fake inputs if needed...
        if inputs is None:
            inputs = tf.zeros([batch_size, sequence_lengths, 1],
                              dtype=tf.float32)
        #inputs.set_shape((None, sequence_lengths, inputs.get_shape()[2]))
        self.states, self.last = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell,
            cell_bw=self.cell,
            dtype=tf.float32,
            # sequence_length = sequence_lengths,
            inputs=inputs,
            initial_state_fw=self.init_fw,
            initial_state_bw=self.init_bw,
        )

        # concatenate the outputs of the encoders (h only) into one vector
        self.last_fw, self.last_bw = self.last

        if rnn_type.lower() == 'lstm':
            self.last_tot = tf.concat(axis=1, values=[self.last_fw.h, self.last_bw.h])
        else:
            self.last_tot = tf.concat(axis=1, values=[self.last_fw, self.last_bw])


class DynamicRNN(object):
    def __init__(self, state_dim, batch_size, name, sequence_lengths,
                 cell=None, inputs=None, initial_state=None, rnn_type='gru',
                 output_size=None, output_size2=None, output_keep_prob=1.0,
                 input_keep_prob=1.0):
        if initial_state is None:
            # need initial states for fw and bw
            self.init_stddev = 1 / np.sqrt(float(state_dim))
            self.init_initter = tf.random_normal_initializer(0.0, self.init_stddev, dtype=tf.float32)

            self.init_h = tf.get_variable(name + '_init_h', [1, state_dim],
                                          initializer=self.init_initter,
                                          dtype=tf.float32)
            if rnn_type.lower() == 'lstm':
                self.init_c = tf.get_variable(name + '_init_c', [1, state_dim],
                                              initializer=self.init_initter,
                                              dtype=tf.float32)

            tile_dimensions = [batch_size, 1]

            self.init_h_tiled = tf.tile(self.init_h,
                                        tile_dimensions, name=name + '_tile')

            if rnn_type.lower() == 'lstm':
                self.init_c_tiled = tf.tile(self.init_c,
                                            tile_dimensions, name=name + '_tile')

            if rnn_type.lower() == 'lstm':
                # tuple for lstm
                self.init = tf.contrib.rnn.LSTMStateTuple(self.init_c_tiled, self.init_h_tiled)
            else:
                self.init = self.init_h_tiled
        else:  # if initial state is None
            self.init = initial_state

        # pick your cell
        if rnn_type.lower() == 'lstm':
            self.cell = tf.nn.rnn_cell.LSTMCell(num_units=state_dim,
                                                state_is_tuple=True)
        elif rnn_type.lower() == 'gru':
            self.cell = tf.nn.rnn_cell.GRUCell(num_units=state_dim)
        else:
            self.cell = cell

        # add dropout if requested
        if output_keep_prob != 1.0:
            self.cell = tf.contrib.rnn.DropoutWrapper(
                self.cell, output_keep_prob=output_keep_prob)

        # unused
        # convenient way to take a low-D output from network
        if output_size is not None:
            self.cell = tf.contrib.rnn.OutputProjectionWrapper(self.cell, output_size=output_size)

        # unused
        # convenient way to take another output from network
        if output_size2 is not None:
            self.cell = tf.contrib.rnn.OutputProjectionWrapper(self.cell, output_size=output_size2)

        # for some reason I can't get dynamic_rnn to work without inputs
        #  so generate fake inputs if needed...
        if inputs is None:
            inputs = tf.zeros([batch_size, sequence_lengths, 1],
                              dtype=tf.float32)
        # call dynamic_rnn
        #inputs.set_shape((None, sequence_lengths, inputs.get_shape()[2]))
        self.states, self.last = tf.nn.dynamic_rnn(
            cell=self.cell,
            dtype=tf.float32,
            # sequence_length = sequence_lengths,
            inputs=inputs,
            initial_state=self.init,
        )


class LinearTimeVarying(object):
    # self.output = linear transform
    # self.output_nl = nonlinear transform

    def __init__(self, inputs, output_size, transform_name, output_name, nonlinearity=None,
                 collections=None, W=None, b=None):
        num_timesteps = tf.shape(inputs)[1]
        # must return "as_list" to get ints
        input_size = inputs.get_shape().as_list()[2]
        outputs = []
        outputs_nl = []
        # use any matrices provided, if they exist
        if W is not None and b is None:
            raise ValueError('LinearTimeVarying: must provide either W and b, or neither')
        if W is None and b is not None:
            raise ValueError('LinearTimeVarying: must provide either W and b, or neither')

        if W is None and b is None:
            W, b = init_linear_transform(input_size, output_size, name=transform_name,
                                         collections=collections)
        self.W = W
        self.b = b

        inputs_permuted = tf.transpose(inputs, perm=[1, 0, 2])
        initial_outputs = tf.TensorArray(dtype=tf.float32, size=num_timesteps, name='init_linear_outputs')
        initial_outputs_nl = tf.TensorArray(dtype=tf.float32, size=num_timesteps, name='init_nl_outputs')

        # keep going until the number of timesteps
        def condition(t, *args):
            return t < num_timesteps

        def iteration(t_, output_, output_nl_):
            # apply linear transform to input at this timestep
            ## cur = tf.gather(inputs, t, axis=1)
            # axis is not supported in 'gather' until 1.3
            # cur = tf.gather(inputs_permuted, t)
            cur = inputs_permuted[t_, :, :]
            output_this_step = tf.matmul(cur, self.W) + self.b
            output_ = output_.write(t_, output_this_step)
            if nonlinearity is 'exp':
                output_nl_ = output_nl_.write(t_, tf.exp(output_this_step))
            return t_ + 1, output_, output_nl_

        i = tf.constant(0)
        t, output, output_nl = tf.while_loop(condition, iteration, \
                                             [i, initial_outputs, initial_outputs_nl])

        self.output = tf.transpose(output.stack(), perm=[1, 0, 2])
        self.output_nl = tf.transpose(output_nl.stack(), perm=[1, 0, 2])

        ## this is old code for the linear time varying transform
        # was replaced by the above tf.while_loop

        # for step_index in range(tensor_shape[1]):
        #    gred = inputs[:, step_index, :]
        #    fout = tf.matmul(gred, self.W) + self.b
        #    # add a leading dimension for concatenating later
        #    outputs.append( tf.expand_dims( fout , 1) )
        #    if nonlinearity is 'exp':
        #        nlout = tf.exp(fout)
        #        # add a leading dimension for concatenating later
        #        outputs_nl.append( tf.expand_dims( nlout , 1) )
        # concatenate the created list into the factors
        # self.output = tf.concat(outputs, axis=1, name=output_name)
        # if nonlinearity is 'exp':
        #    self.output_nl = tf.concat(outputs_nl, axis=1, name=output_name)


class KLCost_GaussianGaussian(object):
    """log p(x|z) + KL(q||p) terms for Gaussian posterior and Gaussian prior. See
    eqn 10 and Appendix B in VAE for latter term,
    http://arxiv.org/abs/1312.6114

    The log p(x|z) term is the reconstruction error under the model.
    The KL term represents the penalty for passing information from the encoder
    to the decoder.
    To sample KL(q||p), we simply sample
          ln q - ln p
    by drawing samples from q and averaging.
    """

    def __init__(self, z, prior_z):
        """Create a lower bound in three parts, normalized reconstruction
        cost, normalized KL divergence cost, and their sum.

        E_q[ln p(z_i | z_{i+1}) / q(z_i | x)
           \int q(z) ln p(z) dz = - 0.5 ln(2pi) - 0.5 \sum (ln(sigma_p^2) + \
              sigma_q^2 / sigma_p^2 + (mean_p - mean_q)^2 / sigma_p^2)

           \int q(z) ln q(z) dz = - 0.5 ln(2pi) - 0.5 \sum (ln(sigma_q^2) + 1)

        Args:
          zs: posterior z ~ q(z|x)
          prior_zs: prior zs
        """
        # L = -KL + log p(x|z), to maximize bound on likelihood
        # -L = KL - log p(x|z), to minimize bound on NLL
        # so 'KL cost' is postive KL divergence
        kl_b = 0.0
        # for z, prior_z in zip(zs, prior_zs):
        assert isinstance(z, Gaussian)
        assert isinstance(prior_z, Gaussian)
        # ln(2pi) terms cancel
        kl_b += 0.5 * tf.reduce_sum(
            prior_z.logvar - z.logvar
            + tf.exp(z.logvar - prior_z.logvar)
            + tf.square((z.mean - prior_z.mean) / tf.exp(0.5 * prior_z.logvar))
            - 1.0, [1])

        self.kl_cost_b = kl_b
        self.kl_cost = tf.reduce_mean(kl_b)


class Poisson(object):
    """Poisson distributon

    Computes the log probability under the model.

    """

    def __init__(self, log_rates):
        """ Create Poisson distributions with log_rates parameters.

        Args:
          log_rates: a tensor-like list of log rates underlying the Poisson dist.
        """
        self.logr = log_rates

    def logp(self, bin_counts):
        """Compute the log probability for the counts in the bin, under the model.

        Args:
          bin_counts: array-like integer counts

        Returns:
          The log-probability under the Poisson models for each element of
          bin_counts.
        """
        k = tf.to_float(bin_counts)
        # log poisson(k, r) = log(r^k * e^(-r) / k!) = k log(r) - r - log k!
        # log poisson(k, r=exp(x)) = k * x - exp(x) - lgamma(k + 1)
        return k * self.logr - tf.exp(self.logr) - tf.lgamma(k + 1)


"""Wrappers for primitive Neural Net (NN) Operations."""

import numbers
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


# dropout that also returns the binary mask
def dropout(x, keep_prob, noise_shape=None, seed=None, name=None,
            binary_tensor=None):  # pylint: disable=invalid-name
    """Computes dropout.
    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.
    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.
    Args:
      x: A floating point tensor.
      keep_prob: A scalar `Tensor` with the same type as x. The probability
        that each element is kept.
      noise_shape: A 1-D `Tensor` of type `int32`, representing the
        shape for randomly generated keep/drop flags.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      name: A name for this operation (optional).
    Returns:
      A Tensor of the same shape of `x`.
    Raises:
      ValueError: If `keep_prob` is not in `(0, 1]` or if `x` is not a floating
        point tensor.
    """
    with ops.name_scope(name, "dropout", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)

        # Only apply random dropout if mask is not provided

        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)

        keep_prob = keep_prob if binary_tensor is None else 1 - keep_prob

        keep_prob = ops.convert_to_tensor(keep_prob,
                                          dtype=x.dtype,
                                          name="keep_prob")

        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if binary_tensor is None:
            # Do nothing if we know keep_prob == 1
            if tensor_util.constant_value(keep_prob) == 1:
                return x, None

            noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
            # uniform [keep_prob, 1.0 + keep_prob)
            random_tensor = keep_prob
            random_tensor += random_ops.random_uniform(noise_shape,
                                                       seed=seed,
                                                       dtype=x.dtype)
            # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            binary_tensor = math_ops.floor(random_tensor)
        else:
            # check if binary_tensor is a tensor with right shape
            binary_tensor = math_ops.cast(binary_tensor, dtype=x.dtype)
            # pass

        ret = math_ops.div(x, keep_prob) * binary_tensor
        # if context.in_graph_mode():
        ret.set_shape(x.get_shape())
        return ret, binary_tensor
