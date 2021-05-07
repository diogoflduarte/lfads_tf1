from helper_funcs import DiagonalGaussianFromExisting, LearnableDiagonalGaussian, diag_gaussian_log_likelihood
from helper_funcs import KLCost_GaussianGaussian, KLCost_GaussianGaussianProcessSampled, LearnableAutoRegressive1Prior
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()



#np.random.seed(10)
tf.random.set_random_seed(10)
bs = 300
dim = 10
T = 200

inp_m = tf.random.normal([bs, T, dim], mean=0.0, stddev=0.1, seed=10)
inp_v = tf.random.normal([bs, T, dim], mean=0.0, stddev=1.0, seed=11)

pr_var = 0.1


post = DiagonalGaussianFromExisting(inp_m, inp_v)

#print(post.mean[:,0,:])

autocorrelation_taus = [0.0001 for _ in range(dim)]
noise_variances = [pr_var for _ in range(dim)]
ar_prior = LearnableAutoRegressive1Prior(bs, dim, autocorrelation_taus, noise_variances, do_train_prior_ar_atau=False, do_train_prior_ar_nvar=False, name= 'ar_prior')
kl_obj = KLCost_GaussianGaussianProcessSampled(post, ar_prior)


#diag_prior = LearnableDiagonalGaussian(bs, [T,dim], 'diagprior', pr_var)
#kl_obj = KLCost_GaussianGaussian(post, diag_prior)




print('lfadslite KL cost {}, sum {}'.format(tf.reduce_mean(kl_obj.kl_cost_b,0), np.sum(kl_obj.kl_cost_b)))


