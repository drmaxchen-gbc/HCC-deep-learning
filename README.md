# HCC
 <br/>
Details for this paper named "Classification and mutation prediction based on histopathology H&E images in liver cancer using deep learning"

------------------------------------------------------------------------------------------------------

## Contents

### Prerequisites
### Data
####     >	Whole slide images
####     >	Crop into “Tiles” and convert into jpg 
####     >	Sort “Tiles” into training/test/internal_validation/external_validation sets and put them into appropriate classes
### Training and testing/validating (choose anyone of following methods)
####     >	Method 1: Training model using codes 
####     >	Method 2: Use EASY DL without any codes, which based on the similar Algorithm with Method 1 (For freshman or non-computer specialists and readers)
### Performance of the model  
### Performance evaluation

--------------------------------------------------------------------------------------------------------

### Prerequisites

	Python (3.6) <br/>
	Numpy (1.14.3)<br/>
	Scipy (1.0.1)<br/>
	[PyTorch (0.3.1)/CUDA 8.0](https://pytorch.org/previous-versions/) The specific binary wheel file is [cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl.](http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl)I have not tested on other versions, especially 0.4+, wouldn't recommend using other versions.<br/>
	torchvision (0.2.0)<br/>
	PIL (5.1.0)<br/>
	scikit-image (0.13.1)<br/>
	[OpenSlide 3.4.1](https://openslide.org/)(Please don't use 3.4.0 as some potential issues found on this version)/[openslide-python (1.1.0)](https://github.com/openslide/openslide-python)<br/>
	matplotlib (2.2.2)<br/>
Most of the dependencies can be installed through pip install with version number, <br/>
e.g.<br/>
``` 
pip install 'numpy==1.14.3'
```
For PyTorch please consider downloading the specific wheel binary and use
```
pip install torch-0.3.1-cp36-cp36m-linux_x86_64.whl
```

### Data
