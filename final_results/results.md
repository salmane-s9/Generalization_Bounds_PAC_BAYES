#### Computation of bound with/without torch.no_grad():

| nb_monte_crl_apprx  | nb_images  | torch.no_grad() | time,s  |
|---   | ---| ---| ---|
|5 | 64 | no| 137.51|
|5 | 64 | yes| 137.51|
|10|100|no|222.01|
|10|100|yes|218.42|
| 20|100|no| 437.2|
|20|100|yes|436.6|

Almost no difference in final bound's computational time, which uses monte-carlo approximation,
with usage of torch.no_grad().

#### torch.parallel

Can be used to parallelize training by distributing batches over available GPUs if many. Since, we don't use batches in this implementation, it becomes useless.

 #### Computation of Bound with different batch sizes and nb_Montecarlo_approx = 10, CUDA_FALSE

| batch_size | time,s  |
|------------| ---------|
|1| 328.95781540870667|
|100| 100.56375193595886|
|5000| 92.92931532859802 |
|10000| 107.97015833854675|
|20000| 89.8711404800415|
|30000|  90.85719680786133|
|55000|  89.74113273620605|

#### Computation of Bound with different batch sizes and nb_Montecarlo_approx = 10, CUDA_TRUE
| batch_size | time,s  |
|------------| ---------|
|10000| 62.72370648384094 |
|55000|  56.778838872909546|

