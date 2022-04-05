
# An Illustrative Guide to Masked Image Modelling

Masked image modelling can provide competitive results to the other approaches like contrastive learning. Performing computer vision tasks using masked images can be called masked image modelling.

In machine learning, nowadays, we can see that the models and techniques of one domain can perform tasks of other domains. For example, models focused on natural language processing can also perform a few tasks related to computer vision. In this article, we will discuss such a technique that is transferable from NLP to computer vision. When applying it to the computer vision tasks, we can call it Masked Image Modelling. We will try to understand the working of this technique along with its important applications.

## What is Masked Image Modelling?

In machine learning, masked signal learning is a type of learning where the masked portion of the input is used to learn and predict the masked signal. We can find the use cases of this type of learning in NLP for self-supervised learning. In many works, we can see the use of masked signal modelling for learning from huge unannotated data. While talking about the computer vision task, this approach can also provide competitive results to the other approaches like contrastive learning. Performing computer vision tasks using masked images can be called masked image modelling.

Applying masked image modelling can have the following difficulties:

- Pixels near to each other are highly correlated.
- Signals under the images are raw and low level in comparison to the signal (tokens) under the NLP data.(CV中低语义)
- Signals in image data are continuous while text signals are discrete.

So applying this approach to image or computer vision-related data, requires the procedure to be accomplished very well so that correlation can be avoided. Prediction from the low-level signals can be used for high-level visual tasks and the approach can adapt the continuous signal behaviour. 

We can witness various works which have modelled image data to generalize these difficulties like:

- [Pre-Trained Image Processing Transformer](https://arxiv.org/pdf/2012.00364.pdf): This work shows the adoption of continuous signal from image for classification tasks using the colour clustering techniques in addition. 
- [Swin Transformer V2](https://arxiv.org/abs/2111.09883): This work represents the technique for scaling a Swin transformer up to 3 billion parameters and making it capable of learning and performing computer vision tasks with images up to 1536 x 1536 resolution. They have applied the adaptation techniques for continuous signals from images using models.
- [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254): This work can be considered as using the BERT model in computer vision where we can witness a similar technique of tokenization using an additional network for image data and its block-wise image masking breaks the short-range connection between the pixels. 

In the above image, we can see the input patches of images, with a linear layer to perform regression on pixel values of the masked area under loss. The design and insights of a simple model can consist of the following:

- Masking applied to the images
- Model for raw pixel regression
- Lightweight prediction head

By applying simple masking to the images, we can make the process simple for a transformer. The regression task aligns well with the continuous nature of visual signals, and a lightweight prediction head should have the property of bringing a remarkable speedup in pre-training. Heavier heads have the capability of a stronger generation but can lead to a loss in the downstream fine-tuning tasks.

## The framework of Masked Image Modelling

We could understand that the motive of these procedures is to learn representation using the masked image modelling, in which the procedure should be capable of masking a portion of an image signal and predicting the original signals at the masked area. A framework to complete the motivation can have the following components:

- Masking strategy: This component should be designed for selecting the area to mask and to perform the masking on the selected area so that the masked image can be used as an input.
- Encoder architecture: This component should be able to extract latent feature representation for the masked image and use the extracted representation to predict the original signals at the masked area. If using transformers as encoders, then it is expected from the encoder that it should be capable of performing a variety of computer vision tasks. Some examples of transformers in computer vision are vanilla ViT and Swim transformers.
- Prediction head: This component should be capable of producing one form of original signals at the masked area of the image when applied to the latent feature representation learned by the encoder. 
- Prediction target: This component should be capable of defining the form of the prediction from original signals and loss type. Talking about the prediction from it can be either raw pixels or a transformation of the raw pixels. The form of loss type can be a cross-entropy classification or L1 and L2 regression loss. 

For image masking, we can use a variety of strategies of image masking like square shape masking, block-wise masking, random masking, etc. The below image is a representation of the different types of image masking.

## Works Related to Masked Image Modelling

Some of the important works related to masked image modelling are as follows:

- [Generative Pretraining from Pixels](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V1_ICML.pdf): In this work, we can see an example of a trained sequence transformer to predict pixels using an autoregressive approach. In this work, a GPT-2 model is used to learn strong image representations. This work has achieved 96.3% accuracy with a linear probe, outperforming a supervised Wide ResNet, and 99.0% accuracy with full fine-tuning, matching the top supervised pre-trained models.
- [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192): In this work, we can see the use of spatial context from the image as a source of visual signals which can be used for training a visual representation. While extracting random pairs of patches from images, they have trained a CNN to predict the place of the second patch correlated to the first. This work is a representation of unsupervised visual discovery by learning feature representations within images. Also, learning feature representations within images helps in capturing the visual similarity across images.
- [Selfie: Self-supervised Pre Training for Image Embedding](https://arxiv.org/abs/1906.02940): This work generalizes the concept of masked language modelling of BERT to image data. Using the masked image, the method learns to select the correct patch of the image, among other patches sampled from the same image. This work on ImageNet 224 x 224 with 60 examples per class (5%), improves the mean accuracy of ResNet-50 from 35.6% to 46.7%, an improvement of 11.1 points in absolute accuracy.
- [SimMIM: A Simple Framework for Masked Image modelling](https://arxiv.org/pdf/2111.09886.pdf): SimMIM is a very simple framework for masked image modelling. This framework is an example of applying a very light prediction head that can be compared to the linear layers. Using ViT-B, this approach achieves 83.8% top-1 fine-tuning accuracy on the ImageNet-1K dataset. By pre-training also on this dataset using SwinV2-H, it achieves 87.1% accuracy using only ImageNet-1K data.
- [Masked image modelling with Autoencoders](https://keras.io/examples/vision/masked_image_modeling/): This is an example of masked image modelling given by Keras where we can find a simple and effective method to pre-train large vision models like ViT. This method gets inspiration from the pre-training algorithm of BERT. Using this example we can learn how to patch images and predict by learning from extracted patches of images. This can be considered as an implementation of a masked autoencoder for self-supervised pre-training with the CIFAR-10 data.

## Applications of Masked Image Modelling

Performing image masking helps transformers and autoencoders to learn easily using only required information from the images. Masking can speed up the transformer to perform classification tasks using images. Also, masking images is a process of creating an image piece from a larger image and also we can use it to modify a larger image. It is a process that is underneath many types of image processing like edge detection, motion detection, and noise reduction. Mainly, we can say that this technique can be used in self-supervised learning in computer vision. Masked images are easy to learn because of the low and important information in masked images. Due to high-level unannotated data creating confusion for the model, image masking can be considered as a process of converting high dimensional data to a lower dimension. 