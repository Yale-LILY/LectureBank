## This is the repo for LectureBank2.0
Check our paper [R-VGAE: Relational-variational Graph Autoencoder for Unsupervised Prerequisite Chain Learning](https://arxiv.org/abs/2004.10610) COLING, 2020.


# data
The dataset, 322 proposed topics and the prerequisite annotations can be found in the folder of `data`. 

`322topics_final.tsv`: the topic list.

`final_new_annotation.csv`: the annotation of the prerequisites.

`lecturebank20.tsv`: *Lecturebank 2.0* dataset. Similar format with 1.0 version. Due to some regulations, you need to download the data using the URL links we provided. We could not distribute plain texts. 

# code
`gae_directed`: The code is built upon [pygcn code (Pytorch)](https://github.com/tkipf/pygcn) and [vgae code (Pytorch)](https://github.com/zfjsail/gae-pytorch).
Obviously you will need to process the data and load the annotated relations. But the code only provides a basic idea on the R-GAE model. 

