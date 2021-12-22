**Notes regarding tutorial (Jakob)**

Week 1:
  * Implemented PSNR - Peak Signal to Noise Ratio for the training and validation set
  * Split up the notebook to the basic and advanced one
  * Implemented saving the model state 
  * Implemented outputting the images to a file (for running on the cluster) - changed visualizing function
  * Made runnable on the cluster (ran for 2.5h for the basic one and 1.5h for the advanced one)
  * Trying label smoothing

Week 2:
  * model with more channels
  * model with different lr
  * model with lsgan
  * model with pretraining

Week 3:
  * still trying to make the model better than the original basic one
  * trying to train the current best model on cartoon dataset
  * combined the stuff that improved the model into one
  * tested label smoothing for real and facke label
  * tested kaiming weight init

Week 4:
  * run advanced model for comp
  * run best model so far (combined) with cartoon images
  * improving the combined one
  
**Notes regarding Instance-aware Image Colorization (Denis)**

Week 1:
  * Implemented using random 8k images form the Coco dataset
  * Ran the code for detectron (2h runtime GPU) - saved the bounding boxes to file (so we don't have to run it again)
  * Fixed repo cloning issue
  * Mount the google drive on colab to save the checkpoints there even if the runtime disconnects
  * problem: the initial code uses 150+150+30 epochs while the paper says 2+5+2 epochs (takes 35 min for 2 epochs with batchsize 1)

Week 2:
  * Have ran the instance model and the full image model 
  * Working on getting the fusion model to work
  * problem: some images have less than 8 detected bounding boxes which the code does not like -> current fix is batch size 1 or using 1 object box
  * Putting the fusion model on the cluster to train (using 1 box for training)
  
Week 3:
  * Make PSNR work for the instance aware model
  * The trained instance aware model does not give okay results (consider not using it). The model did not colorize a lot of example images, and only a few had weird pixels.
  * 

Week 4:
  * Trying to train fusion module with the initial options


**Notes regarding comparisons (Kristin)**
  * Made a visualization function for testing/comparing different tutorial model versions
  * Tested the (current best) model trained on real world images on a few cartoon/anime images
  * Found a cartoon dataset to use for training (train on cartoon and test on real for example)
