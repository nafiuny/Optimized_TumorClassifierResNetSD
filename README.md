# Optimized_TumorClassifierResNetSD
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


## Option1: Download and Preprocess the dataset
If you want to download the dataset and perform the preprocessing yourself, you can use the following code.<br/>
The first command will download the dataset from Google Drive, and the second one will process the data into the required format.<br/>
```
!python download_dataset.py
```
```
!python preprocess.py \
        --train_dir Brain_Tumor_MRI_Dataset/Training \
        --test_dir Brain_Tumor_MRI_Dataset/Testing \
        --output_dir outputs/data

```

## Option2: Use preprocessed files
If you prefer to use the preprocessed files without running the preprocessing step, you can download the preprocessed files (train_data.pt, val_data.pt, and test_data.pt) using the following code.<br/>
Once the preprocessed files are downloaded, you can move directly to training the model.<br/>
```
!python download_preprocessed_data.py
```

## Dataset
Finally, the structure of the 'data' folder where the preprocessed files are placed is as follows.<br/>
```
/outputs/data
├── test_label.pt
├── test_data.pt
├── train_label.pt
├── train_data.pt
├── test_label.pt
├── test_data.pt
```  



## Train
Train Optimizied_TumorClassifierResNetSD.
<br/>
You can modify the values of the input parameters as per your requirements. Alternatively, you can run the script with the default settings without changing any parameters.
```
!python train.py \
        --model_name "resnet_sd" \
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
## Test

```
!python test.py \
        --model_name "resnet_sd" \
        --checkpoint_name "model_resnet_sd" \
        --test_data_path "outputs/data/test_data.pt" \
        --test_labels_path "outputs/data/test_labels.pt" 
```
