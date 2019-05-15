# Free VQA Models from Knowledge Inertia By Pairwise Inconformity Learning.(PIL)

Tensorflow implementation for PIL.

## Citation
If you find PIL useful in your research, please consider citing:

```
@inproceedings{AAAI: 19,
Author={Yiyi Zhou, Rongrong Ji, Jinsong Su, Xiangming Li ,Xiaoshuai Sun},
Title={Free VQA Models from Knowledge Inertia By Pairwise Inconformity Learning},
Booktitle={Thirty-Third AAAI Conference on Artificial Intelligence},
Year={2019}}
```

## Running Code

In this code, you can run our model on VQA2.0 dataset. The code has been tested by Python 2.7, [tensorflow 1.8.0] and CUDA 9.0 on Ubuntu 16.04.

## Preprocessing


In terms of the baseline model, we use the Glove Embedding as the word input with a dimension of 300. The dimension of the LSTM module is 2048, while the k and o in MFB fusion are set to 5 and 1000, respectively. The dimensions of the last forward layer and the projections are set to 2048 and 300. The initial learning rate is 7e-4, which is halved after every 25,000 steps. The batch size is 64 and the maximum training step is set to 150,000. The optimizer we used is Adam.
During experiments, we use two types of visual inputs, i.e., the last feature map of ResNet-152 with a size of 14*14*2048 and the regional features released by with a size of 36*2048. For simplicity, we denote them as CNN and FRCNN.(The code we currently provide is using the CNN feature). Before training, you should put the features in DATA and then update the path in the config.py file.

## Running Example

### Train

To train a model, edit the corresponding config.py file.

In config.py, the TRAIN_DATA_SPLITS, QUESTION_VOCAB_SPACE, and ANSWER_VOCAB_SPACE parameters take a + delimited string of data sources, which are specified in the DATA_PATHS dictionary. We recommend using train+val for training.Remember to set GPU_ID and path correctly.

Now just run python train.py. Training can take some time. Snapshots are saved according to the settings in config.py. To stop training, just hit Control + C.

## Tips

If you find any problems, please feel free to contact to the authors (xiangmingxml@foxmail.com).
