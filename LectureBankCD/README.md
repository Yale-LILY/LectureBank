# LectureBankCD Dataset

This is the repo for our ACL 2021 paper **Unsupervised Cross-Domain Prerequisite Chain Learning using Variational Graph Autoencoders**.

If you use the data or code, please cite our paper:

    @inproceedings{li2021unsupervised,
	Author = {Irene Li and Vanessa Yan and Tianxiao Li and Rihao Qu and Dragomir Radev},
	Title = {Unsupervised Cross-Domain Prerequisite Chain Learning using Variational Graph Autoencoders},
	Booktitle  = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)},
	Year = {2021}
    }
    
    
## Data
Similar format data with LectureBank2 and LectureBank under `Data` folder.
Specifically, we provide spliting train/test/val for both CV and BIO domain under `BenchmarkData` folder. In each domain subfolder, we provide tuples of (`train.<fold>.tsv`,`test.<fold>.tsv`,`val.<fold>.tsv`). Our paper report mean results on these 5 splitings. 

## Code
Stay tuned!
