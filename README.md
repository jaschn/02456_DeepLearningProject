# DeepLearningProject - Image colorization

## Goals
Compare two different approaches for colorizing images. A [conditional adversarial network based model](https://arxiv.org/abs/2005.10825) which is often used as a baseline and a new [instance-aware image colorization approach](https://arxiv.org/abs/1611.07004). We also test different modifications to the baseline model. As a final experiment all models which are trained on real world images are later applied to artificial cartoon images to compare performance.

---
## Data

The models were trained on 2 different datasets: 

* Subset of the cocostuff dataset - 10 000 images (80/20 split)
* A cartoon image dataset found on Kaggle - 4445 images (80/20 split)

The images are resized, flipped horizontally for data augmentation and finally converted into Lab color space. The first (grayscale) channel and the color channels are separated to become inputs and targets for the models respectively. For the instance-aware model, images go through an object-detection model to get bounding boxes, then they are cropped out and resized.

---
## Metrics
The main metric used to compare model performances was PSNR (Peak Signal to Noise Ratio). We also conduct a visual comparison of the output images.

---
## Experiments

* Trying different modifications of the baseline model (e.g. generator pretraining, label smoothing, changing learning rates)
* Training the best modified baseline model on the cartoon images
* Transfer learning the instance-aware model on the cartoon images
* Testing all the models on both real and artificial images

--- 
## Results
### Results of real images

![Results1](/Comparisons/results_real.PNG)

### Results on artificial images

![Results2](/Comparisons/results_artificial.PNG)

