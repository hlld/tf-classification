# TensorFlow Classification

Implementation of TensorFlow-2.x training and testing classification pipeline

## Development History

* `2021-08-01` - support training pipeline
* `2021-07-31` - support custom datasets
* `2021-07-30` - support resnet series

## Requirements
```
pip install -r requirements.txt
```

## Getting Start Training

### Training ResNet50 on ILSVRC2012 dataset
```
python train.py               \
    --data_root='path'        \
    --model_type='resnet50'   \
    --epochs=90               \
    --batch_size=256
```
- The ILSVRC2012 dataset needs to be prepared before training. For details, see [examples/imagenet](https://github.com/pytorch/examples/tree/master/imagenet)

### Fine-tuning ResNet50 on CUSTOM dataset
```
python train.py               \
    --data_root='path'        \
    --model_type='resnet50'   \
    --weights='pre-trained'   \
    --epochs=30               \
    --batch_size=128
```
- The custom dataset format should be consistent with ILSVRC2012. More specially, the reference format of a custom dataset is as follows:
```
dataset_name
├── test
│   ├── class_name_1
│   │   └── image.jpg
│   ├── class_name_2
│   │   └── image.jpg
│   └── class_name_3
│       └── image.jpg
├── train
│   ├── class_name_1
│   │   └── image.jpg
│   ├── class_name_2
│   │   └── image.jpg
│   └── class_name_3
│       └── image.jpg
└── val
    ├── class_name_1
    │   └── image.jpg
    ├── class_name_2
    │   └── image.jpg
    └── class_name_3
        └── image.jpg
```

### Training with multi-GPU or multi-machine
```
python train.py --device='0,1,2,3'
```

## About The Author

A boring master student from CQUPT. Email `hlldmail@qq.com`
