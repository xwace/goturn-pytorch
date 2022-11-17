# main script to train

IMAGENET_PATH='/home/star/Desktop/goturn-pytorch/datasets/ImageNet/'
ALOV_PATH='/home/star/Desktop/goturn-pytorch/datasets/ALOV/'
SAVE_PATH='./caffenet/'
PRETRAINED_MODEL_PATH='/home/star/Desktop/goturn-pytorch/src/goturn/models/pretrained/caffenet_weights.npy'

python3 train.py \
--imagenet_path $IMAGENET_PATH \
--alov_path $ALOV_PATH \
--save_path $SAVE_PATH \
--pretrained_model $PRETRAINED_MODEL_PATH
