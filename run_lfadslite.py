import tensorflow as tf
import numpy as np
import sys
import time


# utils defined by CP
#from helper_funcs import linear, init_linear_transform, makeInitialState
#from helper_funcs import DiagonalGaussianFromInput, DiagonalGaussian, DiagonalGaussianFromExisting
#from helper_funcs import BidirectionalDynamicRNN, DynamicRNN, LinearTimeVarying
#from helper_funcs import KLCost_GaussianGaussian, Poisson
#from complexcell import ComplexCell

from helper_funcs import ListOfRandomBatches, kind_dict
from plot_funcs import plot_data, close_all_plots
from data_funcs import read_datasets


from lfadslite import LFADS



data_dir = '/tmp/rnn_synth_data_v1.0/'
#data_fn = 'chaotic_rnn_inputs_g1p5_dataset_N50_S50'
data_fn = 'chaotic_rnn_inputs_g1p5_dataset'
#data_dir = '/home/cpandar/th/lf2/lorenz/' # lorenz data
#data_fn = 'generated_data'
#data_fn = 'generated'
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


hps = {}
hps['num_steps'] = train_data.shape[1]
hps['dataset_dims'] = {}
hps['dataset_dims'][dkey] = train_data.shape[2]
hps['batch_size'] = 80
hps['sequence_lengths'] = [hps['num_steps'] for i in range(hps['batch_size'])]














# hardcode some HPs for now
#networks
hps['ic_enc_dim'] = 64
hps['gen_dim'] = 64
hps['ci_enc_dim'] = 64
hps['con_dim'] = 64
# inputs to the controller from the ci_enc
#  NEW in lfadslite: this allows you to restrict dimensionality
hps['con_ci_enc_in_dim'] = 10 
hps['con_fac_in_dim'] = 10

#ics, cis, factors
hps['factors_dim'] = 20
hps['ic_dim'] = 10
hps['co_dim'] = 1

hps['cell_clip_value'] = 5.0
hps['do_causal_controller'] = True

# hps
hps['keep_prob']=0.95
hps['ic_prior_var']=0.1
hps['ic_post_var_min']=0.0001
hps['co_prior_var']=0.1
hps['co_post_var_min']=0.0001
hps['kind'] = 'train'
hps['max_grad_norm'] = 200.0
hps['learning_rate_init'] = 0.05
hps['learning_rate_decay_factor'] = 0.97
hps['learning_rate_n_to_compare'] = 6
hps['learning_rate_stop'] = 0.00001

hps['kl_ic_weight'] = 1.0
hps['kl_co_weight'] = 1.0

##   haven't implemented these in lfadslite
# hps['feedback_factors_or_rates'] = 'factors'







# initialize the model with these hyperparams
model = LFADS(hps)











# define an epoch
epoch_batches = []
for npochs in range(200):
    epoch_batches += ListOfRandomBatches(train_data.shape[0], hps['batch_size'])

valid_batches = ListOfRandomBatches(valid_data.shape[0], hps['batch_size'])

steps = range(0,len(epoch_batches))
steps_per_epoch = np.floor(train_data.shape[0] / hps['batch_size'])

#print_trainable_vars(trainable_vars)
#sys.exit()


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
            tc_e = []; rc_e = []; kc_e = []
        
        # train the model on this batch
        X = train_data[epoch_batches[nstep],:,:].astype(np.float32)
        feed_dict = {'input_data': X,
                     'keep_prob': hps['keep_prob']}
        evald = model.train_batch(feed_dict)
        [_, tc, rc, kc, rts, lr] = evald
        # print out the step cost
        print 'step ' + str(nstep) +': [ '+ str(tc) +', ' + str(rc) + '], \r',
        sys.stdout.flush()
        #store costs for the epoch
        tc_e.append(tc); rc_e.append(rc); kc_e.append(kc)
        
        writer.close()
        
        # should we decrement learning rate
        n_lr = hps['learning_rate_n_to_compare']
        if len(train_costs) > n_lr and tc > np.max(train_costs[-n_lr:]):
            print("Decreasing learning rate")
            model.run_learning_rate_decay_opt()

        train_costs.append(tc)
            
        new_lr = model.get_learning_rate()
        # should we stop?
        if new_lr < hps['learning_rate_stop']:
            print("Learning rate criteria met")
            break


        #run the validation set once per epoch
        if nstep % steps_per_epoch == 0:
            # returns the total cost, reconstruction cost, and kl cost
            tcv_all = []; rcv_all = []; kcv_all = []
            for nvbatch in range(len(valid_batches)):
                Xv = valid_data[valid_batches[nvbatch],:,:].astype(np.float32)
                feed_dict = {'input_data': Xv}
                [tcv_b, rcv_b, kcv_b] = model.validation_batch(feed_dict)
                tcv_all.append(tcv_b); rcv_all.append(rcv_b); kcv_all.append(kcv_b)

            # take the mean of all validation batches
            tcv = np.mean(tcv_all); rcv = np.mean(rcv_all); kcv = np.mean(kcv_all)
            # take the mean of all training batches            
            tct = np.mean(tc_e); rct = np.mean(rc_e); kct = np.mean(kc_e)
            print
            print "epoch: %i, step %i: total: (%f, %f), rec: (%f, %f), kl: (%f, %f), lr: %f" % (nepoch, nstep, tct, tcv, rct, rcv, kct, kcv, lr)
            
            if hps['co_dim'] > 0:
                feed_dict = {'input_data': X}
                [cos] = model.get_controller_output(feed_dict)
                cos1 = cos[0,:,:]
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
