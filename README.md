# speedlines-detection

Abstract: This repo includes a pipeline for training UNet with different encoders to detect compressor speedlines. Moreover, trained model are provided LINK
Dataset already explained in task text and it seems okay. So i do not see any obstacles not to skip EDA.

## Plan of research
First, let's identify the main architecture. UNet is a bit better for this problem than Mask R-CNN. It is enough to complete the task, without the need to use more complex instance segmentation like Mask R-CNN. I've conducted a research on several Kaggle kernels and papers from sources like arxiv.com.

So:
 - Architecture: UNet
 - Encoder: EfficientNet-B0,B1; ResNet-34
 - Loss function: FocalLoss (alpha = 0.8 gamma = 2), DiceLoss, bce_jaccard_loss
 - Optimizer: Adam (lr = 1e-3)
 - learning scheduler: ReduceLROnPlateau(factor=0.5, patience=5)

## General thoughts

I've tried DiceLoss, bce_jaccard_loss, FocalLoss
The best results have been obtained with bce_jaccard_loss in this case.

All of the encoders were pretrained on ImageNet. However, I do believe there is **one more trick** that can be fruitful: we can fine-tune encoders on the whole dataset (classification combined/separated). This way we can get some better results, but there is no structured dataset at all.

Moreover, we can try some multi-scale training methods to increase image resolution from small to large, but I haven't done that.

I need to add I've been bounded with Cuda memory capacity, so basicaly I could not try bigger encoders for batch size > 16.
val_iou_score: 0.9212 - val_dice_metric: 0.9590

## Results

| Encoder | IoU | dice_metric | Mask Resolution | Epochs |
| ------ | ------ | ------ | ------ | ------ |
| ResNet-34 | 0.9078 |  0.9516   | (256, 256) |       50     |
| EfficientNet-B0  | 0.9253 | 0.9612 | (256, 256)| 50 |
| EfficientNet-B1  | 0.9212 | 0.9590 | (256, 256) |    50    |



Inferences for validation data:

Combined case:


 - EfficientNet-B0

 Example 1:
 ![alt text](/imgs/pic_(1).jpg)

 Example 2: 
 ![alt text](/imgss/pic_(4).jpg)
 
 Example 3: 
 ![alt text](/imgs/pic_(2).jpg)
 
 Example 4: 
 ![alt text](/imgs/pic_(9).jpg)
 
Simple case:
 
  Example 1:
 ![alt text](/imgs/pic_(13).jpg)

 Example 2: 
 ![alt text](/imgs/pic_(11).jpg)
 
 Example 3: 
 ![alt text](/imgs/pic_(8).jpg)
 
