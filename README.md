# Optimizied_TumorClassifierResNetSD
A Novel Optimized ResNet-SD Model for Brain Tumor Classification Using Stochastic Depth and Metaheuristic Optimization


<p align="justify">
This is the official <strong>Optimizied_TumorClassifierResNetSD</strong> implementation repository with PyTorch.<br/><br/>

</p>
<p align="center">
<br><br><br><br>
<img src="imgs/Optimizied_TumorClassifierResNetSD.png" width="500">
<br>
<b>Optimizied TumorClassifierResNetSD Architecture</b>
<br><br><br><br>
</p>

## Setup

Clone the repository.

```
!git clone https://https://github.com/nafiuny/Optimized_TumorClassifierResNetSD.git
%cd Optimized_TumorClassifierResNetSD
```

## preprocess


```
!python preprocess.py \
  --train_dir Brain_Tumor_MRI_Dataset/Training \
  --test_dir Brain_Tumor_MRI_Dataset/Testing \
  --output_dir outputs/data
```


## Train

Train Optimizied_TumorClassifierResNetSD
```
!python train.py --model_name "resnet_sd" \
                 --checkpoint_name "model_resnet_sd" \
                 --train_data_path "outputs/data/train_data.pt" \
                 --train_labels_path "outputs/data/train_labels.pt" \
                 --val_data_path "outputs/data/val_data.pt" \
                 --val_labels_path "outputs/data/val_labels.pt" \
                 --num_epochs 200 \
                 --lr 0.01 \
                 --stochastic_depth1 0.6 \
                 --stochastic_depth2 0.7 \

```

