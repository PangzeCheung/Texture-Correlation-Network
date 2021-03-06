# Texture Correlation Network
The source code for our paper "[Lightweight Texture Correlation Network for Pose Guided Person Image Generation](https://ieeexplore.ieee.org/abstract/document/9631236)“, Pengze Zhang, Lingxiao Yang, Xiaohua Xie and Jianhuang Lai, TCSVT 2021.


## Get Start

### Train
#### DeepFashion
Stage1:
``` 
python train.py --name TCN_fashion --model PoseAlign --checkpoints_dir ./checkpoints --dataset_mode fashion --dataroot XXXX/Fashion --batchSize 64 
```
Stage2:
``` 
python train.py --name TCN_fashion --model TCN --checkpoints_dir ./checkpoints --dataset_mode fashion --dataroot XXXX/Fashion --lambda_style 1000 --lambda_content 1 --lambda_lg1 2 --lambda_lg2 5 --lambda_g 2 --batchSize 12 --continue_train
``` 

#### Market
``` 
python train.py --name TCN_market --model TCN --checkpoints_dir ./checkpoints --dataset_mode market --dataroot XXXX/Market --lambda_style 500 --lambda_content 0.5 --lambda_lg1 1 --lambda_lg2 2 --lambda_g 5 --batchSize 64
``` 

### Test
#### DeepFashion
``` 
python test.py --name TCN_fashion --model TCN --checkpoints_dir ./checkpoints --dataset_mode fashion --dataroot XXXX/Fashion
``` 
#### Market
``` 
python test.py --name TCN_market --model TCN --checkpoints_dir ./checkpoints --dataset_mode market --dataroot XXXX/Market 
``` 


## Acknowledgement 

We build our project base on (https://github.com/RenYurui/Global-Flow-Local-Attention & https://github.com/daa233/generative-inpainting-pytorch). Some dataset preprocessing methods are derived from (https://github.com/tengteng95/Pose-Transfer).

