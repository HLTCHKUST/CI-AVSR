# CI-AVSR
Repository for the paper [CI-AVSR: A Cantonese Audio-Visual Speech Dataset for In-car Command Recognition](https://arxiv.org/pdf/2201.03804.pdf) and the corresponding new dataset, which is accepted in LREC 2022.

If you find our dataset or code useful, please cite this paper, thanks!
<pre>
@article{Dai2022CIAVSRAC,
  title={CI-AVSR: A Cantonese Audio-Visual Speech Dataset for In-car Command Recognition},
  author={Wenliang Dai and Samuel Cahyawijaya and Tiezheng Yu and Elham J. Barezi and Peng Xu and Cheuk Tung Shadow Yiu and Rita Frieske and Holy Lovenia and Genta Indra Winata and Qifeng Chen and Xiaojuan Ma and Bertram E. Shi and Pascale Fung},
  journal={ArXiv},
  year={2022},
  volume={abs/2201.03804}
}
</pre>

# Data 
***Version 1.0***

For details of the dataset, please refer to the paper.

### Clean Sets
The originally collected data, including `train_clean.csv`, `valid_clean.csv`, `test_clean.csv`.

### Noise Augmented Sets
The noise augmented data, including `train_noisy.csv`, `valid_noisy.csv`, `test_noisy.csv`. In addition, there is also a out-of-domain test set `test_noisy_ood.csv` to evaluate the generalization of models, i.e. these noises are not in the training set.

### Data Download
[[CI-AVSR at Kaggle]](https://www.kaggle.com/c/ci-avsr/data) We provide the processed CI-AVSR dataset with audios, image frames (25 per seconds), annotations, and augmentation. 
