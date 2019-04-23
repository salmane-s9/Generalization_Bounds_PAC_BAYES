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
