# Learning Localized Generative Models for 3D Point Clouds via Graph Convolution (ICLR 2019)

Warning: trained models are large! A code-only repository is available at: https://github.com/diegovalsesia/GraphCNN-GAN-codeonly

If you like our work, please cite the journal version of the paper.

Journal version BibTex reference:
```
@ARTICLE{Valsesia2019journal,
author={Diego {Valsesia} and Giulia {Fracastoro} and Enrico {Magli}},
journal={under review},
title={Learning Localized Representations of Point Clouds with Graph-Convolutional Generative Adversarial Networks},
year={2019},
volume={},
number={},
pages={},
}
```

ICLR 2019 BibTex reference:
```
@inproceedings{valsesia2019learning,
  title={Learning Localized Generative Models for 3D Point Clouds via Graph Convolution},
  author={Valsesia, Diego and Fracastoro, Giulia and Magli, Enrico},
  booktitle={International Conference on Learning Representations (ICLR) 2019},
  year={2019}
}
```

# Requirements

  - Python 2.7
  - Tensorflow >=1.6 

# Usage
  
  A trained model for the method with aggregation upsampling is provided for the following Shapenet classes: airplane, chair, sofa, table.
  - launch_test.sh : generate a batch of point clouds from the specified class 
  - launch_train.sh : retrain the network (requires downloading the Shapenet dataset and place it in the data directory)
