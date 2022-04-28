# handgesture-music
## Description
My bachelor project. I trained a deep learning model to recognize 8 hand gestures and mapped it to a media player to play music with the control of gestures as an interface.
The gap between static image classificaion and real-time classification was bridged successfully. In the experimentation, 36 models were trained on 8 architectures that vary in depth, number of output feature maps, activation functions and other changes. The highest performing model achieved 99 percent on the training and test set with great performance in real-time classification

## Architecture of highest performing model 
* 7 layers with 1,731,816 parameters
* No dense layers before classification instead max global pooling
* Fully convolutional
* used reflective padding to overcome boundries' effect
* used Mish activation function
* dropout and batch normalization after each convolution

## Training schedule for the best model
* cross entropy loss function
* Adam optimizer
* 0.0008 learning rate
* 32 batch size
* 100 epochs
* augmentation includes:
	*  multiple color augmentations (colorjitter, fancyPCA, CLAHE, etc.),
	* deformation (grid distortion) 
	* noise augmentations (blue, ISO noise, etc.).
	
**NOTE:** models were not included in the files due to large size.
