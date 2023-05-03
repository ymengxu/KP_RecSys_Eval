# Evaluating recommender systems on Adobe and MIND dataset
This project aims to provide an evaluation of the performance of different recommender systems on the KP click dataset (adobe) and general-domain interaction data. 

## Dataset
- adobe: [source](https://confluence-aes.kp.org/display/RecSys/Research+Datasets+for+RecSys+building+and+evaluation) (~14GB one) stored in data_raw/adobe
- MIND: [source](https://msnews.github.io/) stored in data_raw/mind (TODO)
- MovieLens-100k: [source](https://grouplens.org/datasets/movielens/) stored in data_raw/movielens/ml-100k

## Algorithms
- [iALS](http://yifanhu.net/PUB/cf.pdf), [BPR](https://arxiv.org/pdf/1205.2618.pdf): pure matrix factorization methods designed for implicit data
- [LightFM](https://arxiv.org/pdf/1507.08439.pdf): hybrid matrix factorization
- [tensor](https://arxiv.org/pdf/1905.02009.pdf): hybrid tensor factorization
- [NCF](https://arxiv.org/pdf/1708.05031.pdf): (hybrid) matrix factorization
- YoutubeNet/others: two-tower model
- [LightGCN](https://arxiv.org/pdf/2002.02126.pdf): GCN-based CF
- [BiVAE](https://dl.acm.org/doi/abs/10.1145/3437963.3441759)
- [Wide&Deep](https://arxiv.org/abs/1606.07792): linear and neural based CF
- news recommenders (designed to utilize rich text information) (TODO)
	- [MKN](https://arxiv.org/abs/1901.08907)
	- [MIECL](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_699.pdf) 

## Metrics
precision, recall, HR, NDCG, MRR, MAP

## Contact
Repo Link: https://github.com/ymengxu/KP_RecSys_Eval
Repo Link in KP: https://github.kp.org/CSIT-CDO-KPWA/RecSys_Algos_Eval
Author: Mengxuan Yan (mengxuan.yan@alumni.emory.edu)