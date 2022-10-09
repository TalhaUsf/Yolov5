#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com

import argparse


parser = argparse.ArgumentParser()



# # --------------------------------------------------------------------------
# #                              related to data                        
# # --------------------------------------------------------------------------
parser.add_argument(
    "--train_annotations_dir",
    type=str,
    default="/home/fsuser/AI_ENGINE/yolov5_tf_original/Yolov5/data/voc_train.txt",
    help="train annotations path",
)
parser.add_argument(
    "--test_annotations_dir",
    type=str,
    default="/home/fsuser/AI_ENGINE/yolov5_tf_original/Yolov5/data/voc_test.txt",
    help="test annotations path",
)
parser.add_argument(
    "--class_name_dir",
    type=str,
    default="/home/fsuser/AI_ENGINE/yolov5_tf_original/Yolov5/data/sample/voc.names",
    help="classes name path",
)
parser.add_argument(
    "--yaml_dir", type=str, default="/home/fsuser/AI_ENGINE/yolov5_tf_original/Yolov5/yolo/configs/yolo-m-mish.yaml", help="model.yaml path"
)
# # --------------------------------------------------------------------------
# #                              related to training                        
# # --------------------------------------------------------------------------


parser.add_argument("--log_dir", type=str, default="/home/fsuser/AI_ENGINE/yolov5_tf_original/Yolov5/logs", help="log path")
parser.add_argument(
    "--checkpoint_dir", type=str, default="/home/fsuser/AI_ENGINE/yolov5_tf_original/Yolov5/weights/weight-{epoch}", help="saved checkpoint path"
)
parser.add_argument(
    "--saved_model_dir",
    type=str,
    default="/home/fsuser/AI_ENGINE/yolov5_tf_original/Yolov5/weights/yolov5",
    help="saved pb model path",
)

parser.add_argument("--n_epochs", type=int, default=130)
parser.add_argument(
    "--batch_size", type=int, default=32, help="total batch size for all GPUs"
)
parser.add_argument("--multi_gpus", type=bool, default=True)
parser.add_argument("--init_learning_rate", type=float, default=1e-3)
parser.add_argument("--warmup_learning_rate", type=float, default=1e-6)
parser.add_argument("--warmup_epochs", type=int, default=2)
parser.add_argument("--img_size", type=int, default=640, help="image target size")
parser.add_argument("--mosaic_data", type=bool, default=False, help="if mosaic data")
parser.add_argument("--augment_data", type=bool, default=True, help="if augment data")
parser.add_argument(
    "--anchor_assign_method", type=str, default="wh", help="assign anchor by wh or iou"
)
parser.add_argument(
    "--anchor_positive_augment",
    type=bool,
    default=True,
    help="extend the neighbour to positive",
)
parser.add_argument(
    "--label_smoothing", type=float, default=0.02, help="classification label smoothing"
)

args = parser.parse_args()
params = vars(args)
