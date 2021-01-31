# DIP-Project

**Train model:**

```shell
cd models/research/object_detection

python model_main_tf2.py --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config --model_dir=training --alsologtostderr
```

**Some notes for the presentation:**

- Splitted up data into 80% train and 20% test.
- Improved background of all images by dividing the original background before feeding data into network.
- Labeling was done manually by 3 classes: Indiana Jones, Unknown Guy, Unknown Girl.
- Additional labels gave us more possibilities for the evaluation process.
- Additional labels gave us less wrong predictions.
- We used a [tensorflow 2 - model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) of a network which can be used as a pre-trained version on the [COCO 2017-dataset.](https://cocodataset.org/#home)
- We trained this model from scratch for our own purpose (to detect LEGO figures).
- The COCO dataset offers a huge collection of images, with segmented objects. It's a perfect dataset for all kinds of object detection.
- Installed cuda and cudNN drivers for gpu-based training.
- We configured a config file before the training process
- 50k training steps have been done, which took us around 5 hours.
- With Tensorboard you can watch the training process live (all kinds of loss factors).
- We converted our model to a frozen graph file, so the prediction functionality can be easily called from python

**Useful links:**

- [Tensorflow 2 Object Detection Library](https://blog.roboflow.com/the-tensorflow2-object-detection-library-is-here/)
- [EfficientDet](https://blog.roboflow.com/breaking-down-efficientdet/)