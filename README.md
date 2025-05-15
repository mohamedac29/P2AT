## P2AT: Pyramid Pooling Axial Transformer for Real-time Semantic Segmentation  [[Arxiv]](https://arxiv.org/abs/2310.15025)
## The paper has been accepted at Expert Systems with Applications [ESWA](https://www.sciencedirect.com/science/article/abs/pii/S0957417424014775)

You need to download the Cityscapesdatasets. and rename the folder cityscapes, then put the data under data folder.

### Datasets Preparation

## 1. Cavmvid Dataset
You can download the [Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camvid)

## Citysscapes Dataset
* You need to download the [Cityscapes](https://www.cityscapes-dataset.com/) datasets, unzip them and put the files in the `data` folder with following structure.

```
$SEG_ROOT/data\ 
├── Camvid
│       ├── images
│       ├── labels

│    ├── cityscapes
│        ├── gtFine
│            ├── test
│            ├── train
│            ├── val
│     ── ── leftImg8bit
│             ├── test
│             ├── train
│             └── val
│    ├── list
         ├── Camvid
│          ├── test.lst
│          ├── train.lst
│          ├── trainval.lst
│          └── val.lst
│       ├── cityscapes
│          ├── test.lst
│          ├── train.lst
│          ├── trainval.lst
│          └── val.lst
   
```

### Training

##  Training on Camvid datsaset

* For instance, train the P2AT-S on Camvid dataset with batch size of 8 on 2 GPUs:
````bash
python train.py --cfg configs/camvid/p2at_small_camvid.yaml GPUS (0,1) TRAIN.BATCH_SIZE_PER_GPU 4
````


