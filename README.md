# speedlines-detection

Abstract: This repo includes a pipeline for training UNet with different encoders to detect compressor speedlines. Moreover, trained model are provided LINK
Dataset already explained in task text and it seems okay. So i do not see any obstacles not to skip EDA.

## Plan of research
First, let's identify the main architecture. UNet is a bit better for this problem than Mask R-CNN. It is enough to complete the task, without the need to use more complex instance segmentation like Mask R-CNN. I've conducted a research on several Kaggle kernels and papers from sources like arxiv.com.

So:
 - Architecture: UNet
 - Encoder: EfficientNet-B0,B1; ResNet-50
 - Loss function: FocalLoss (alpha = 0.8 gamma = 2), DiceLoss, bce_jaccard_loss
 - Optimizer: Adam (lr = 1e-3)
 - learning scheduler: ReduceLROnPlateau(factor=0.5, patience=5)
 
 B0+UNET shuffle FocalLoss

Epoch 00049: val_loss did not improve from 0.00104
Epoch 50/50
200/200 [==============================] - 53s 265ms/step - loss: 0.0028 - iou_score: 0.3847 - dice_metric: 0.5549 - val_loss: 0.0010 - val_iou_score: 0.5553 - val_dice_metric: 0.7139

Epoch 00049: val_loss did not improve from 0.00103
Epoch 50/50
200/200 [==============================] - 54s 270ms/step - loss: 0.2617 - iou_score: 0.7623 - dice_metric: 0.8648 - val_loss: 0.0834 - val_iou_score: 0.9251 - val_dice_metric: 0.9611

Epoch 00050: val_loss did not improve from 0.00103

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=sm.losses.bce_jaccard_loss, metrics = [sm.metrics.iou_score, dice_metric] )

Epoch 00049: val_loss did not improve from 0.03790
Epoch 50/50
160/160 [==============================] - 43s 270ms/step - loss: 0.1304 - iou_score: 0.7699 - dice_metric: 0.8696 - val_loss: 0.0391 - val_iou_score: 0.9253 - val_dice_metric: 0.9612

Epoch 00050: val_loss did not improve from 0.03790



## General thoughts

I've tried DiceLoss, bce_jaccard_loss, FocalLoss
The best results have been obtained with bce_jaccard_loss in this case.

All of the encoders were pretrained on ImageNet. However, I do believe there is **one more trick** that can be fruitful: we can fine-tune encoders on the whole dataset (classification combined/separated). This way we can get some better results, but there is no structured dataset at all.

Moreover, we can try some multi-scale training methods to increase image resolution from small to large, but I haven't done that.

I need to add I've been bounded with Cuda memory capacity, so basicaly I could not try bigger encoders for batch size > 16.

## Results

| Encoder | IoU | DiceBCELoss | Mask Resolution | Epochs |
| ------ | ------ | ------ | ------ | ------ |
| ResNet-50 | 0.4132 |     | (256, 1600) |            |    |
| EfficientNet-B3  | 0.513  |      0.444        | (256, 768)| 11 |
| EfficientNet-B4  | 0.597 | 0.36 | (256, 768) |    37     |



Inferences for validation data:

 - EfficientNet-B4

 Example 1:
 ![alt text](/images/pic_1.png)

 Example 2: 
 ![alt text](/images/pic_2.png)
 
 Example 3: 
 ![alt text](/images/pic_3.png)
 
 Example 4: 
 ![alt text](/images/pic_4.png)
