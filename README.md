# Explanation_by_Progressive_Exaggeration




## Usage
1. Download the CelebA dataset and create train and test fold for the classifier. 

```
notebooks/Data_Processing.ipynb
```
2. Train a classifier. Skip this step if you have a pretrained classifier.

2.a. To train a multi-label classifier on all 40 attributes
```
python train_classifier.py --config 'configs/celebA_DenseNet_Classifier.yaml'
```
2.b. To train a binary classifier on all 1 attribute
```
python train_classifier.py --config 'configs/celebA_Smile_DenseNet_Classifier.yaml'
```
3. Process the output of the classifier and create input for Explanation model by discretizing the posterior probability.

```
python main.py --cls_experiment 'output/CelebA/Center_Crop/Smiling/Smiling_DenseNet'
```
