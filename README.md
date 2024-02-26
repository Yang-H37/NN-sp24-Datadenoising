# Part 1: Conceptual design
## Data
**Training and Validation DataSets:**

A collection of photos taken with natural environmental noise. They are split into a 3:1 ratio for training and validation.

**Test DataSet:**

A set of photos taken by myself in low-light conditions and provide natural environmental noise. (If the quantity is insufficient, a portion of the training and validation sets will be further divided to augment the test set.)

## Model
The structure of an autoencoder consists of an encoder and a decoder. The encoder encodes input images with noise to obtain a feature representation of the image.For image feature extraction, available encoders include convolutional neural networks (CNN), self-attention networks, etc.

**CNN:** 

CNNs are a popular choice for image feature extraction due to their ability to capture features. Comprising multiple convolutional layers, pooling layers, and fully connected layers, CNNs extract features and reduce parameters through convolutional and pooling layers. Furthermore, techniques such as residual connections can be incorporated to enhance feature extraction capabilities, enabling the model to effectively denoise images while preserving important details.

**Self-Attention Networks:**

Self-attention mechanisms enable the model to capture long-range dependencies within the input data, making them particularly suitable for image processing tasks. By dynamically adjusting attention weights based on the input data's context, self-attention networks can effectively highlight relevant features for denoising. Additionally, integrating multi-head attention mechanisms and positional encoding can enhance the network's ability to extract informative features from noisy images.

After obtaining the feature representation of the image through the encoder, the decoder reconstructs the image to minimize noise impact and ensure clear generation. Available decoders include deconvolutional networks, generative adversarial networks (GANs), etc.

**Deconvolutional Networks:**

Deconvolutional networks consist of a series of deconvolutional layers and upsampling layers. These networks gradually upsample feature maps while performing convolution operations, ultimately reconstructing denoised images with reduced noise artifacts.

**Generative Adversarial Networks(GANS):**

GANs employ a generative model and a discriminative model in a competitive training framework. The generative model, often referred to as the decoder, learns to generate realistic images, while the discriminative model evaluates the authenticity of the generated images. By iteratively improving the generator's ability to produce realistic images and the discriminator's ability to differentiate between real and generated images, GANs can effectively denoise images while giving generated images more realistic.

Additionally, as a variant of autoencoders, variational autoencoders encode and decode data into probability distributions. Variational autoencoders and their variants can be employed to construct image denoising models. 

In model construction, it is anticipated to use a self-attention network to construct the encoder and a generative adversarial network to construct the decoder for the image denoising task. Various combinations of the mentioned networks will be explored to seek the optimal solution.

## Training and Evaluation
Employ mean squared error or cross-entropy as the loss function, with Adam or SGD optimizers. Evaluate image denoising effectiveness using metrics like PSNR(Peak Signal-to-Noise Ratio) and SSIM(Structural Similarity Index).

# Part 2: Datasets
## Source
### Description of SIDD dataset
The Smartphone Image Denoising Dataset (SIDD) is a dataset which has nearly 30,000 noisy images from 10 scenes under different lighting conditions, using five representative smartphone cameras and generated their ground truth images. Using the noisy images and their ground truth, we can train and validate our own data denoising algorithms.

Each image is stored in a directory with the name of the scene instance as follows:

**_[scene-instance-number][scene-number][smartphone-camera-code][ISO-level][shutter-speed][illuminat-temperature][illuminant-brightness-code]_**

where "smartphone-camera-code" is one of the following:
+ GP: Google Pixel
+ IP: iPhone 7
+ S6: Samsung Galaxy S6 Edge
+ N6: Motorola Nexus 6
+ G4: LG G4

and 

"illuminant-brightness-code" is one of the following:
+ L: low light
+ N: normal brightness
+ H: high exposure

The SIDD dataset is publicly available and can be downloaded from the official website: 

[SIDD Dataset](https://www.eecs.yorku.ca/~kamel/sidd/index.php).

The dataset is introduced in the following paper: 

[Abdelrahman Abdelhamed, Lin S., Brown M. S. "A High-Quality Denoising Dataset for Smartphone Cameras", IEEE Computer Vision and Pattern Recognition (CVPR), June 2018.](https://ieeexplore.ieee.org/document/8578280)

## Differences Between Train and Validation Subsets
In ideal daytime shooting conditions, the ISO value of the camera should be kept around 100-200. However, in special conditions such as dusk or darkness, it is necessary to increase the ISO value of the camera to enhance its shooting capability in low-light conditions. In such cases, the ISO parameter usually increases to 1600 or higher. Increasing the ISO value introduces more noise, which is the main problem to be addressed during image denoising in smartphone photography.

Therefore, when dividing the dataset, we will use 60% of the data from the SIDD dataset as training data. In the training set, we will focus on selecting images from different shooting scenes, with higher ISO values (greater than or equal to 1600), and lower shooting brightness, to enable the model to learn more about image denoising in dark scenes and form more reasonable weight and bias learning parameters. Additionally, to ensure the generalization of the model, about 25% of the training set data will be added, including images from different shooting scenes, lower ISO values, and normal or higher shooting brightness.

Furthermore, we will use 20% of the data from the SIDD dataset as the validation set. To prevent overfitting, the proportion of images in the validation set from different shooting scenes, with higher ISO values (greater than or equal to 1600), and lower shooting brightness to images with lower ISO values, normal or higher shooting brightness will be adjusted to 1:3. This adjustment increases the number of images in normal shooting environments and prevents the model from overfitting to enhance the denoising effect on dark condition images.

## Distinct objects/subjects
There are a total of 160 scenes, with 41 having an ISO value greater than or equal to 1600 and 119 having an ISO value less than 1600. Each scene has a sample size of 150.

## Brief Characterization of Samples
+ resolution: 4032×3024
+ sensors used: GP: Google Pixel, IP: iPhone 7, S6: Samsung Galaxy S6 Edge, N6: Motorola Nexus 6, G4: LG G4 (smartphone camera)
+ ambient conditions: low light, normal brightness, high exposure
+ ISO: 100, 200, 400, 500, 640, 800, 1600, 3200, 6400, 10000
