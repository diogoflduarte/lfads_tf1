lfadslite
============

lfadslite is a stripped down implementation of LFADS[1]. The primary motivation was to redesign the underlying architecture so it is compatible with Tensorflows "dynamic_rnn" and bidirectional_dynamic_rnn functions, which greatly reduce memory usage while also speeding up execution.

Several pieces of the original LFADS architecture are not (yet?) implemented, notably, multi-session, continuous (gaussian) output, L2 penalities on the recurrent weights, gradual stepping of the regularizers, AR priors, observed external inputs (i.e., not from the controller, data jittering, weight scaling.