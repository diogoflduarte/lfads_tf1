import tensorflow as tf
import numpy as np
import sys
import time

# utils david defined
#from distributions import LearnableDiagonalGaussian, DiagonalGaussianFromInput

# utils i've defined
from helper_funcs import linear, init_linear_transform, makeInitialState
from helper_funcs import ListOfRandomBatches, kind_dict
from helper_funcs import DiagonalGaussianFromInput, DiagonalGaussian, DiagonalGaussianFromExisting
from helper_funcs import BidirectionalDynamicRNN, DynamicRNN, LinearTimeVarying
from helper_funcs import KLCost_GaussianGaussian, Poisson
from plot_funcs import plot_data, close_all_plots
from data_funcs import read_datasets
from complexcell import ComplexCell


tf.reset_default_graph();

# hardcode some HPs for now
hps = {}
#networks
hps['ic_enc_dim'] = 64
hps['gen_dim'] = 64
hps['ci_enc_dim'] = 64
hps['con_dim'] = 64
hps['con_enc_in_dim'] = 10
hps['con_fac_in_dim'] = 10

#ics, cis, factors
hps['factors_dim'] = 20
hps['ic_dim'] = 10
hps['co_dim'] = 1
#hps['feedback_factors_or_rates'] = 'factors' #haven't implemented this

# hps
hps['keep_prob']=0.95
hps['ic_var_min']=0.1
hps['co_var_min']=0.1
hps['batch_size'] = 80
hps['kind'] = 'train'
hps['max_grad_norm'] = 200.0
# hps['cell_clip_value'] = 5.0  #haven't implemented this
hps['learning_rate_init'] = 0.01
hps['learning_rate_decay_factor'] = 0.97
hps['learning_rate_n_to_compare'] = 6
hps['learning_rate_stop'] = 0.00001


# Create input data
#X = np.random.randn(hps['batch_size'], NUM_TIMESTEPS, DATA_DIM).astype(np.float32)
# The second example is of length 6
#X[1,6:] = 0
#X_lengths = [100, 6, 100, 100, 100, 100, 100, 100, 100, 100]


data_dir = '/tmp/rnn_synth_data_v1.0/'
data_fn = 'chaotic_rnn_inputs_g1p5_dataset_N50_S50'
#data_dir = '/home/cpandar/th/lf2/lorenz/' # lorenz data
#data_fn = 'generated_data'
datasets = read_datasets(data_dir, data_fn)
dkey = datasets.keys()[0]
#print datasets[dkey].keys()
#sys.exit()

train_data = datasets[dkey]['train_data']
valid_data = datasets[dkey]['valid_data']
train_truth = datasets[dkey]['train_truth']
valid_truth = datasets[dkey]['valid_truth']

# train_data = train_data[0::5,:,:]
# valid_data = valid_data[0::5,:,:]
# train_truth = train_truth[0::5,:,:]
# valid_truth = valid_truth[0::5,:,:]

print train_data.shape
print valid_data.shape

NUM_TIMESTEPS = train_data.shape[1]
DATA_DIM = train_data.shape[2]
sequence_lengths = [NUM_TIMESTEPS for i in range(hps['batch_size'])]

pullinds = range(0, hps['batch_size'])
Xbatch = train_data[pullinds,:,:]
Xbatch = Xbatch.astype(np.float32, copy=False)

learning_rate = tf.Variable(float(hps['learning_rate_init']), trainable=False, name="learning_rate")
learning_rate_decay_op = learning_rate.assign(learning_rate * hps['learning_rate_decay_factor'])


### BEGIN MODEL CONSTRUCTION

with tf.variable_scope('placeholders'):
    input_data = tf.placeholder(tf.float32, shape = [hps['batch_size'], NUM_TIMESTEPS, DATA_DIM])
    keep_prob = tf.placeholder(tf.float32)
    run_type = tf.placeholder(tf.int16)

with tf.variable_scope('ic_enc'):
    ## ic_encoder
    ic_enc_rnn_obj = BidirectionalDynamicRNN(state_dim = hps['ic_enc_dim'],
                                         inputs = input_data,
                                         sequence_lengths = sequence_lengths,
                                         batch_size = hps['batch_size'],
                                         name = 'ic_enc',
                                         rnn_type = 'gru',
                                         output_keep_prob = keep_prob)

    # map the ic_encoder onto the actual ic layer
    ics_posterior = DiagonalGaussianFromInput(x = ic_enc_rnn_obj.last_tot,
                                z_size = hps['ic_dim'],
                                name = 'ic_enc_2_ics',
                                var_min = hps['ic_var_min'],
                                )

# to go forward, either sample from the posterior, or take mean
#     (depending on current usage)
if run_type in [kind_dict("train"), kind_dict("posterior_sample_and_average")]:
    ics = ics_posterior.sample()
else:
    ics = ics_posterior.mean
with tf.variable_scope('generator'):
    g0 = linear(ics, hps['gen_dim'], name='ics_2_g0')


### CONTROLLER construction
# this should only be done if a controller is requested
# if not, skip all these graph elements like so:
if hps['co_dim'] !=0:
    with tf.variable_scope('ci_enc'):
        ## ci_encoder
        ci_enc_rnn_obj = BidirectionalDynamicRNN(state_dim = hps['ci_enc_dim'],
                                         inputs = input_data,
                                         sequence_lengths = sequence_lengths,
                                         batch_size = hps['batch_size'],
                                         name = 'ci_enc',
                                         rnn_type = 'gru',
                                         output_keep_prob = keep_prob)

    ci_enc_rnn_states_fwd, ci_enc_rnn_states_bw = ci_enc_rnn_obj.states
    ci_enc_rnn_states = tf.concat(ci_enc_rnn_obj.states, axis=2, name='ci_enc_rnn_states')
    
    ## controller inputs
    with tf.variable_scope('ci_enc_2_co_in'):
        # one input from the encoder, another will come back from factors
        controller_enc_inputs_object = LinearTimeVarying(inputs=ci_enc_rnn_states,
                                   output_size = hps['con_enc_in_dim'],
                                   transform_name = 'ci_enc_2_co_in',
                                   output_name = 'controller_enc_input_concat',
                                   nonlinearity = None)
        controller_enc_inputs=controller_enc_inputs_object.output

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
    comcell_init_state = [con_init_state, g0,
                          co_mean_init_state, co_logvar_init_state,
                          fac_init_state]
    complexcell_init_state = tf.concat(axis=1, values = comcell_init_state)


    # complexcell also requires inputs - (thankfully there is only one input)
    complexcell_inputs = controller_enc_inputs

    # here is what the state vector will look like
    comcell_state_dims = [hps['con_dim'],
                          hps['gen_dim'],
                          hps['co_dim'], # for the controller output means
                          hps['co_dim'], # for the variances
                          hps['factors_dim']]

    # construct the complexcell
    complexcell=ComplexCell(hps['con_dim'],
                            hps['gen_dim'],
                            hps['co_dim'],
                            hps['factors_dim'],
                            hps['con_fac_in_dim'],
                            hps['batch_size'],
                            var_min = hps['co_var_min'],
                            kind=run_type)

    # construct the actual RNN
    complex_outputs, complex_final_state =\
    tf.nn.dynamic_rnn(complexcell,
                      complexcell_inputs,
                      initial_state=complexcell_init_state,
                      time_major=False)

    con_states, gen_states, co_mean_states, co_logvar_states, fac_states =\
    tf.split(complex_outputs,
             comcell_state_dims,
             axis=2)

with tf.variable_scope('rates'):
    ## rates
    rates_object = LinearTimeVarying(inputs=fac_states,
                                   output_size = DATA_DIM,
                                   transform_name = 'factors_2_rates',
                                   output_name = 'rates_concat',
                                   nonlinearity = 'exp')
    # get both the pre-exponentiated and exponentiated versions
    logrates=rates_object.output
    rates=rates_object.output_nl



## calculate the KL cost
# build a prior distribution to compare to
ics_prior = DiagonalGaussian(z_size = [hps['batch_size'], hps['ic_dim']], name='ics_prior', var = hps['ic_var_min'])
cos_prior = DiagonalGaussian(z_size = [hps['batch_size'], NUM_TIMESTEPS, hps['co_dim']], name='cos_prior', var = hps['co_var_min'])
cos_posterior = DiagonalGaussianFromExisting(co_mean_states, co_logvar_states)

# print ics_prior.mean_bxn
# print ics_posterior.mean_bxn
# print cos_prior.mean_bxn
# print cos_posterior.mean_bxn
# tmp = cos_posterior.mean_bxn - cos_prior.mean_bxn
# print tmp
# sys.exit()

# cost for each trial
kl_cost_g0_b = KLCost_GaussianGaussian(ics_posterior, ics_prior).kl_cost_b
kl_cost_co_b = KLCost_GaussianGaussian(cos_posterior, cos_prior).kl_cost_b

# total KL cost
kl_cost_g0 = tf.reduce_mean(kl_cost_g0_b)
kl_cost_co = tf.reduce_mean( tf.reduce_mean(kl_cost_co_b, [1]) )
kl_cost = kl_cost_g0+kl_cost_co

## calculate reconstruction cost
loglikelihood_b_t = Poisson(logrates).logp(input_data)
# cost for each trial
log_p_b = tf.reduce_sum( tf.reduce_sum(loglikelihood_b_t, [2] ), [1] )
# total rec cost
log_p = tf.reduce_mean( log_p_b, [0])
rec_cost = -log_p

## calculate total cost
total_cost = kl_cost + rec_cost

# get the list of trainable variables
trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
gradients = tf.gradients(total_cost, trainable_vars)
gradients, grad_global_norm = tf.clip_by_global_norm(gradients, hps['max_grad_norm'])
opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                             epsilon=1e-01)

train_step = tf.get_variable("global_step", [], tf.int64,
                             tf.zeros_initializer(),
                             trainable=False)
train_op = opt.apply_gradients(
    zip(gradients, trainable_vars), global_step=train_step)



# define an epoch
epoch_batches = []
for npochs in range(200):
    epoch_batches += ListOfRandomBatches(train_data.shape[0], hps['batch_size'])

valid_batches = ListOfRandomBatches(valid_data.shape[0], hps['batch_size'])

steps = range(0,len(epoch_batches))
steps_per_epoch = np.floor(train_data.shape[0] / hps['batch_size'])

#print_trainable_vars(trainable_vars)
#sys.exit()

if hps['co_dim'] > 0:
    train_ops_to_eval = [total_cost, rec_cost, kl_cost, train_op, rates, learning_rate]
else:
    train_ops_to_eval = [total_cost, rec_cost, kl_cost, train_op, rates, learning_rate]
valid_ops_to_eval = [total_cost, rec_cost, kl_cost]

# setup tf configuration
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
writer = tf.summary.FileWriter('logs', session.graph)

train_costs = []
nepoch = 0
with session.as_default():
    tf.global_variables_initializer().run()
    for nstep in steps:
        if nstep % steps_per_epoch == 0:
            # new epoch
            nepoch+=1
            tc_e = []
            rc_e = []
            kc_e = []
        # train the model
        X = train_data[epoch_batches[nstep],:,:].astype(np.float32)
        evald = session.run(train_ops_to_eval, feed_dict = {input_data: X,
                                                            keep_prob: hps['keep_prob'],
                                                            run_type: kind_dict("train")})
        writer.close()
        [tc, rc, kc, to, rts, lr] = evald
        #store costs for the epoch
        tc_e.append(tc)
        rc_e.append(rc)
        kc_e.append(kc)
        
        # should we decrement learning rate
        n_lr = hps['learning_rate_n_to_compare']
        if len(train_costs) > n_lr and tc > np.max(train_costs[-n_lr:]):
            _ = session.run(learning_rate_decay_op)

        train_costs.append(tc)
            
        new_lr = session.run(learning_rate)
        # should we stop?
        if new_lr < hps['learning_rate_stop']:
            print("Learning rate criteria met")
            break


        #run the validation set once per epoch
        if nstep % steps_per_epoch == 0:
            # run the validation set
            tcv_all = []
            rcv_all = []
            kcv_all = []
            for nvbatch in range(len(valid_batches)):
                Xv = valid_data[valid_batches[nvbatch],:,:].astype(np.float32)
                [tcv_b, rcv_b, kcv_b] = session.run(valid_ops_to_eval, feed_dict = {input_data: Xv,
                                                                                    keep_prob: 1.0,
                                                                                    run_type: kind_dict("train")})
                tcv_all.append(tcv_b)
                rcv_all.append(rcv_b)
                kcv_all.append(kcv_b)
                
            tcv = np.mean(tcv_all)
            rcv = np.mean(rcv_all)
            kcv = np.mean(kcv_all)
            tct = np.mean(tc_e)
            rct = np.mean(rc_e)
            kct = np.mean(kc_e)
            print "epoch: %i, step %i: total: (%f, %f), rec: (%f, %f), kl: (%f, %f), lr: %f" % (nepoch, nstep, tct, tcv, rct, rcv, kct, kcv, lr)
            
            if hps['co_dim'] > 0:
                evald = session.run([co_mean_states],
                                    feed_dict = {input_data: X,
                                                 keep_prob: 1.0,
                                                 run_type: kind_dict("train")})
                [cos] = evald
                cos1 = cos[0,:,:]
                # evald2 = session.run([co_mean_states],
                #                      feed_dict = {input_data: X,
                #                                   keep_prob: 1.0,
                #                                   run_type: kind_dict("posterior_mean")})
                # [cos2] = evald
                #                cos3 = cos2[0,:,:]
            else:
                cos1 = None
                
            if nepoch % 1 == 0:
                # plot the current results
                plt = plot_data(X[0,:,:], rts[0,:,:], truth=train_truth[epoch_batches[nstep][0],:,:], cos_sample=cos1)
            if nepoch % 20 == 0:
                close_all_plots()
            
#        if nstep % 10 == 0:
#            print "step %i: total: (%f), rec: (%f), kl: (%f), lr: %f" % (nstep, tc, rc, kc, lr)
        
print("Done training")
