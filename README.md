# Semi-supervised Domain Adaptation on Graphs with Contrastive Learning and Minimax Entropy

This is our implementation for the following paper:

>[Xiao, Jiaren, Quanyu Dai, Xiao Shen, Xiaochen Xie, Jing Dai, James Lam, and Ka-Wai Kwok. "Semi-supervised domain adaptation on graphs with contrastive learning and minimax entropy." arXiv preprint arXiv:2309.07402 (2023).](https://arxiv.org/abs/2309.07402)


## Abstract
Label scarcity in a graph is frequently encountered in real-world applications due to the high cost of data labeling. To this end, semi-supervised domain adaptation (SSDA) on graphs aims to leverage the knowledge of a labeled source graph to aid in node classification on a target graph with limited labels. SSDA tasks need to overcome the domain gap between the source and target graphs. However, to date, this challenging research problem has yet to be formally considered by the existing approaches designed for cross-graph node classification. This paper proposes a novel method called SemiGCL to tackle the graph **Semi**-supervised domain adaptation with **G**raph **C**ontrastive **L**earning and minimax entropy training. SemiGCL generates informative node representations by contrasting the representations learned from a graph's local and global views. Additionally, SemiGCL is adversarially optimized with the entropy loss of unlabeled target nodes to reduce domain divergence. Experimental results on benchmark datasets demonstrate that SemiGCL outperforms the state-of-the-art baselines on the SSDA tasks.

## Environment requirement
The codes can be run with the below packages:
* python == 3.7.9
* torch == 1.7.1+cu101
* numpy == 1.15.4
* networkx == 1.9.1
* scipy == 1.5.4

## Examples to run the codes
* Transfer task D -> A
```
python SemiGCL.py --epochs 30 --lr_cly 0.01 --aggregator_class diffusion --output_dims 1024,64 --cal_ssl --ssl_param 0.1 --target_shot 5 --source_dataset dblpv7 --target_dataset acmv9 --n_samples 20,20 --T 20.0 --alpha_ppr 0.1 --diff_k 20 --batch_size 256 --mme_param 1.0
```

* Transfer task B1 -> B2
```
python SemiGCL.py --epochs 50 --lr_cly 0.01 --aggregator_class diffusion --output_dims 1024,64 --cal_ssl --ssl_param 0.1 --target_shot 5 --source_dataset Blog1 --target_dataset Blog2 --n_samples 30,30 --T 15.0 --alpha_ppr 0.1 --diff_k 30 --batch_size 128 --mme_param 2.0 --is_blog
```

## Citation 
If you would like to use our code, please cite:
```
@article{xiao2023semi,
  title={Semi-supervised domain adaptation on graphs with contrastive learning and minimax entropy},
  author={Xiao, Jiaren and Dai, Quanyu and Shen, Xiao and Xie, Xiaochen and Dai, Jing and Lam, James and Kwok, Ka-Wai},
  journal={arXiv preprint arXiv:2309.07402},
  year={2023}
}
```
