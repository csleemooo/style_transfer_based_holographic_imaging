## Physics-Aware Style Transfer for Adaptive Holographic Reconstruction 

We provied pytorch(python) implementations of **Physics-Aware Style Transfer for Adaptive Holographic Reconstruction**. 
This code was written by **Chanseok Lee**.
Last update: 2025.07.02

### Abstract
Inline holographic imaging presents an ill-posed inverse problem of reconstructing objects’ complex amplitude from recorded diffraction patterns. Although recent deep learning approaches have shown promise over classical phase retrieval algorithms, they often require high-quality ground truth datasets of complex amplitude maps to achieve a statistical inverse mapping operation between the two domains. Here, we present a physics-aware style transfer approach that interprets the object-to-sensor distance as an implicit style within diffraction patterns. Using the style domain as the intermediate domain to construct cyclic image translation, we show that the inverse mapping operation can be learned in an adaptive manner only with datasets composed of intensity measurements. We further demonstrate its biomedical applicability by reconstructing the morphology of dynamically flowing red blood cells, highlighting its potential for real-time, label-free imaging. As a framework that leverages physical cues inherently embedded in measurements, the presented method offers a practical learning strategy for imaging applications where ground truth is difficult or impossible to obtain.

### Training and Inference scheme
<p align = "center">
<img src="/Figures/training.png" width="800" height="400">
</p>

## Run demo
### Environements
- Linux / RTX4090 / CUDA 12.4
### Clone
```
git clone https://github.com/csleemooo/style_transfer_based_holographic_imaging.git
```
Move to the cloned directory
```
cd style_transfer_based_holographic_imaging
```
### Install syste requirements
```
pip install -r requirements.txt
```
### Run test
Test on MNIST dataset
```
python test_field_retrieval_mnist.py --device cpu
```
The reconstructed results of complex-valued object function and object-to-sensor distance of input diffraction patterns will be saved at [SAVE_DIR]/[EXP_name]. defult: ./output/MNIST_test

### Comlex-valued object function retrieval on MNIST dataset 
<p align = "center">
<img src="/Figures/Mnist-result.png" width="800" height="600">
</p>

## Acknwoledgements
This repository is implemented based on repository of [AdaIN](https://github.com/naoto0804/pytorch-AdaIN).
