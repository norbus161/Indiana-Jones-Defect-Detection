# DIP-Project

**Install libs:**

```shell
pip install pyyaml

pip install tensorflow
```

**Train model:**

```shell
cd models/research/object_detection

python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```