## Part1: Conceptual design
# Data
**Training and Validation DataSets:**

A collection of photos taken in low-light conditions with natural environmental noise. They are split into a 3:1 ratio for training and validation.

**Test DataSet:**

A set of photos taken by myself in low-light conditions and provide natural environmental noise. (If the quantity is insufficient, a portion of the training and validation sets will be further divided to augment the test set.)

# Model
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

# Training and Evaluation
Employ mean squared error or cross-entropy as the loss function, with Adam or SGD optimizers. Evaluate image denoising effectiveness using metrics like PSNR(Peak Signal-to-Noise Ratio) and SSIM(Structural Similarity Index).
