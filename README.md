# Free VQA Models from Knowledge Inertia By Pairwise Inconformity Learning.(PIL)

Tensorflow implementation for PIL.


## Abstract

In this paper, we uncover the issue of knowledge inertia in visual question answering (VQA), which commonly exists in most VQA models and forces the models to mainly rely on the question content to “guess” answer, without regard to the visual information. Such an issue not only impairs the performance of VQA models, but also greatly reduces the credibility of the answer prediction. To this end, simply highlighting the visual features in the model is undoable, since the prediction is built upon the joint modeling of two modalities and largely influenced by the data distribution. In this paper, we propose a Pairwise Inconformity Learning (PIL) to tackle the issue of knowledge inertia. In particular, PIL takes full advantage of the similar image pairs with diverse answers to an identical question provided in VQA2.0 dataset. It builds a multi-modal embedding space to project pos./neg. feature pairs, upon which word vectors of answers are modeled as anchors. By doing so, PIL strengthens the importance of visual features in prediction with a novel dynamic-margin based triplet loss that efficiently increases the semantic discrepancies between pos./neg. image pairs. To verify the proposed PIL, we plug it on a baseline VQA model as well as a set of recent VQA models, and conduct extensive experiments on two benchmark datasets, i.e., VQA1.0 and VQA2.0. Experimental results show that PIL can boost the accuracy of the existing VQA models (1.56%-2.93% gain) with a negligible increase in parameters (0.85%-5.4% parameters). Qualitative results also reveal the elimination of knowledge inertia in the existing VQA models after implementing our PIL.


## Citation
If you find PIL useful in your research, please consider citing:

```
@inproceedings{AAAI: 19,
  Author={Yiyi Zhou, Rongrong Ji, Jinsong Su, Xiangming Li ,Xiaoshuai Sun},
  Title={Free VQA Models from Knowledge Inertia By Pairwise Inconformity Learning},
  Booktitle={Thirty-Third AAAI Conference on Artificial Intelligence},
  Year={2019}, Accept}
```

## Running Code

In this code, you can run our model on VQA2.0 dataset. The code has been tested by Python 2.7, [tensorflow 1.8.0] and CUDA 9.0 on Ubuntu 16.04.

## Running Example

### Train

```shell
python train.py
```

## Tips

If you find any problems, please feel free to contact to the authors (xiangmingxml@foxmail.com).
