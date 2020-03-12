# PRE-TRAINED CNNs

## VGG
- using only 3x3 conv.layer filers
- painfully slow to train
- network weights are large (VGG16 - 533MB)

## ResNet
- demonstrating that extremely deep networks can be triened using standerd SGD
- much deeper than VGG16 or VGG19
- using global average pooling => model size is smaller (102MB) for ResNet50 (50 - weight layers)

## InceptionV3
- goal is to act as a multi-level feature extractor by computing 1x1, 3x3, 5x5 convolutions
=> output of these filters are then stacked along the channel dimension before being fed into next layer in the network
- from Google
- size of model/weights - 96MB

## Xception
- extension of Inception which replaces the standard Inception modules with depth wise separable convolutions
- size 91MB

Here are examples of image classification by VGG16:

![koala](koala.png)
![pool_table](pool_table.png)
![soccer_ball](soccer_ball.png)
![sports_car](sports_car.png)
