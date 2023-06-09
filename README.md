# speedlines-detection

Abstract: This repo includes a pipeline for training UNet with different encoders to detect compressor speedlines. Moreover, trained [models](https://drive.google.com/drive/folders/1XjiCYWGAfoECp0RebKn6oFC2-9D_x4mw?usp=sharing) are provided.
Dataset already explained in task text and it seems okay. So i do not see any obstacles not to skip EDA.


## Plan of research
 - Architecture: UNet
 - Encoder: EfficientNet-B0,B1; ResNet-34
 - Loss function: DiceLoss, bce_jaccard_loss
 - Optimizer: Adam (lr = 1e-3)
 - learning scheduler: ReduceLROnPlateau(factor=0.5, patience=5)

## General thoughts

I've tried DiceLoss, bce_jaccard_loss:

The Dice coefficient, or Dice-Sørensen coefficient, is a common metric for pixel segmentation that can also be modified to act as a loss function:

 ![alt text](/imgs/Dice.PNG)
 
bce_jaccard_loss is a loss combination: binary_crossentropy + jaccard_loss.
The best results have been obtained with bce_jaccard_loss in this case.

All of the encoders were pretrained on ImageNet. However, I do believe there is **one more trick** that can be fruitful: we can fine-tune encoders on the whole dataset (classification combined/separated). This way we can get some better results, but there is no structured dataset at all.

The encoder part of the EfficientNet model is deeper than the decoder in our case, which means that the bulk of the computation occurs in the encoder layers. 
Therefore, it may be beneficial to increase the learning rate coefficient for the encoder to speed up the convergence of the model during training
and remove possible gradient decay.

However, using the same image resolution for all versions of the EfficientNet model may not be optimal. This is because increasing the model size and depth can cause the gradients to become more unstable and difficult to propagate through the network, particularly if the image resolution remains constant. As a result, it may be necessary to increase the image resolution as well to maintain a balance between model complexity and training stability.

In the case of the UNet model with EfficientNet-B1/B2/B3 encoders, increasing the image resolution may be necessary to achieve optimal performance
**(because of the main idea of EfficientNet: increasing model size along with resolution)**.

Moreover, we can try some multi-scale training methods to increase image resolution from small to large, but I haven't done that.

I need to add I've been bounded with Cuda memory capacity, so basicaly I could not try bigger encoders for batch size > 16. That's why i have not tried bigger (original) resolutions.


## Results

| Encoder | IoU | dice_metric | Mask Resolution | Epochs |
| ------ | ------ | ------ | ------ | ------ |
| ResNet-34 | 0.9078 |  0.9516   | (256, 256) |       50     |
| EfficientNet-B0  | 0.9253 | 0.9612 | (256, 256)| 50 |
| EfficientNet-B1  | 0.9212 | 0.9590 | (256, 256) |    50    |

Number of epochs was choosen to see if reducing of learning rate can give us better results. I have also used transforms from albumentations to provide more images.
In average, 15 epochs seems enough to train the model. Graphs as example for EfficientNet-B0 encoder are provided:

![alt text](/imgs/Accuracy.PNG)

![alt text](/imgs/Loss.PNG)

![alt text](/imgs/lr.PNG)

Overviewing this graphics we do not see any overfitting. This means our model is stable whether it is validation data or train data.
Reducing learning rate also have not too much influence on our model, but there is also a trick to try changing losses during training: Tversky and Focal-Tversky loss benefit from very low learning rates, of the order 5e-5 to 1e-4. I believe that increasing of batch size and fine-tuning encoders on the whole dataset (classification combined/separated) should have relatively significant impact on our results.


Inferences for validation data (EfficientNet-B0):

 - Combined case:

 Example 1:
 ![alt text](/imgs/pic_(1).jpg)

 Example 2: 
 ![alt text](/imgs/pic_(4).jpg)
 
 Example 3: 
 ![alt text](/imgs/pic_(2).jpg)
 
 Example 4: 
 ![alt text](/imgs/pic_(9).jpg)
 
 - Simple case:
 
  Example 1:
 ![alt text](/imgs/pic_(13).jpg)

 Example 2: 
 ![alt text](/imgs/pic_(11).jpg)
 
 Example 3: 
 ![alt text](/imgs/pic_(8).jpg)
 
