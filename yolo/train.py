#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com

import os
import sys

filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import shutil
import time

import numpy as np
import tensorflow as tf
from configs.config import params
from dataset.load_data import DataLoader
from dataset.read_data import DataReader, transforms
from model.loss import YoloLoss
from model.optimizer import LrScheduler, Optimizer
from model.yolo import Yolo

np.random.seed(1919)
tf.random.set_seed(1949)


class Trainer(object):
    """Trainer class that uses the dataset and model to train
    ⚡uses the custom training loop. Convert it to model.fit form
    
    # Usage
    data_loader = tf.data.Dataset()
    trainer = Trainer(params)
    trainer.train(data_loader)
    """

    def __init__(self, params):
        """Constructor
        :param params: dict, with dir and training parameters
        """
        self.params = params
        if os.path.exists(self.params["log_dir"]):
            shutil.rmtree(self.params["log_dir"])
        self.log_writer = tf.summary.create_file_writer(self.params["log_dir"])
        
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.build_model()

    def build_model(self):
        """Build the model,
        define the training strategy and model, loss, optimizer
        :return:
        """
        # FIXME use 
        if self.params["multi_gpus"]:
            # Synchronous training across multiple replicas on one machine.
            # tf.config.set_visible_devices([], 'GPU') # force it to use CPU
            self.strategy = tf.distribute.MirroredStrategy(devices=None)

        else:
            # 👉 train on a specific GPU
            # Synchronous training across multiple replicas on one machine.
            self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

        with self.strategy.scope():
            # ==========================================================================
            #                   define model inside the strategy scope                                  
            # ==========================================================================
            self.model = Yolo(yaml_dir=self.params["yaml_dir"])
            self.anchors = self.model.module_list[-1].anchors
            self.stride = self.model.module_list[-1].stride
            self.num_classes = self.model.module_list[-1].num_classes

            # # --------------------------------------------------------------------------
            # #                     define loss functions inside scope                        
            # # --------------------------------------------------------------------------
            self.loss_fn = YoloLoss(
                self.model.module_list[-1].anchors,
                ignore_iou_threshold=0.3,
                num_classes=self.num_classes,
                label_smoothing=self.params["label_smoothing"],
                img_size=self.params["img_size"],
            )
            # define optimizer inside scope
            # self.optimizer = Optimizer("adam")()
            self.optimizer = Optimizer("sgd")()

    def train(self, train_dataset, valid_dataset=None, transfer="scratch"):
        """train function
        :param train_dataset: train dataset built by tf.data
        :param valid_dataset: valid dataset build by td.data, optional
        :param transfer: pretrain
        :return:
        """
        # how much steps are there in 1 epoch
        steps_per_epoch = train_dataset.len / self.params["batch_size"]
        # how many steps will be there in total
        self.total_steps = int(self.params["n_epochs"] * steps_per_epoch)
        self.params["warmup_steps"] = self.params["warmup_epochs"] * steps_per_epoch

        with self.strategy.scope():
            # define LR scheduler inside scope
            self.lr_scheduler = LrScheduler(
                self.total_steps, self.params, scheduler_method="cosine"
            )
            # get the model instance
            self.model = self.model(self.params["img_size"])

            ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            ckpt_manager = tf.train.CheckpointManager(
                ckpt, self.params["checkpoint_dir"], max_to_keep=5
            )
            if transfer == "darknet":
                print("Load weights from ")
                model_pretrain = Yolo(self.params["yaml_dir"])(self.params["img_size"])
                model_pretrain.load_weights()
                self.model.get_layer().set_weights()
            elif transfer == "resume":
                print("Load weights from latest checkpoint")
                ckpt.restore(ckpt_manager.latest_checkpoint)
            elif transfer == "scratch":
                print("Train from scratch")
                print(self.model.summary())


        # # --------------------------------------------------------------------------
        # #                        distribute the dataset 🛢️                        
        # # --------------------------------------------------------------------------
        train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)

        
        # # --------------------------------------------------------------------------
        # #                              main training loop 🏃‍♀️                        
        # # --------------------------------------------------------------------------
        for epoch in range(1, self.params["n_epochs"] + 1):
            # # --------------------------------------------------------------------------
            # #                        mini-batches training loop 🏃‍♀️                        
            # # --------------------------------------------------------------------------
            for step, (image, target) in enumerate(train_dataset):
                # 🔥 forward pass step
                loss = self.dist_train_step(image, target)
                print(
                    "=> Epoch {}, Step {}, Loss {:.5f}".format(
                        epoch, self.global_step.numpy(), loss.numpy()
                    )
                )
                with self.log_writer.as_default():
                    tf.summary.scalar("loss", loss, step=self.global_step)
                    tf.summary.scalar("lr", self.optimizer.lr, step=self.global_step)
                self.log_writer.flush()
                
            val_loss_epoch = self.validate(valid_dataset)
            tf.summary.scalar("val-loss", val_loss_epoch, step=self.global_step)
            print(f"EPOCH {epoch} \t val-loss {val_loss_epoch}")

            if epoch % 3 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(
                    "Saving checkpoint for epoch {} at {}".format(epoch, ckpt_save_path)
                )

        self.export_model()

    # @tf.function
    def train_step(self, image, target):
        '''
        how forward pass is done

        Parameters
        ----------
        image : tf.Tensor
            [N, H, W, C]
        target : tf.Tensor
            

        Returns
        -------
        total_loss : tf.Tensor
            iou_loss + conf_loss + prob_loss
        '''        
        with tf.GradientTape() as tape:
            logit = self.model(image, training=True)
            print(f"==============================================")
            print(f"LOGITS\t\t {logit.__len__()}")
            print(f"LOGITS[0]\t {logit[0].shape}")
            print(f"LOGITS[1]\t {logit[1].shape}")
            print(f"LOGITS[2]\t {logit[2].shape}")
            print(f"==============================================")
            print(f"TARGET\t\t {target.__len__()}")
            print(f"TARGET[0]\t {target[0].shape}")
            print(f"TARGET[1]\t {target[1].shape}")
            print(f"TARGET[2]\t {target[2].shape}")
            print(f"==============================================")
            iou_loss, conf_loss, prob_loss = self.loss_fn(target, logit)
            # 🔥 Following line is necessary
            total_loss = iou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        lr = self.lr_scheduler.step()
        self.optimizer.lr.assign(lr)
        self.global_step.assign_add(1)
        return total_loss

    @tf.function
    def dist_train_step(self, image, target):
        """
        Distributed training step, called from inside the mini-batches loop
        """
        with self.strategy.scope():
            # define the loss function
            loss = self.strategy.run(self.train_step, args=(image, target))
            total_loss_mean = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, loss, axis=None
            )
            return total_loss_mean

    def validate(self, valid_dataset):
        valid_loss = []
        for step, (image, target) in enumerate(valid_dataset):
            step_valid_loss = self.valid_step(image, target)
            valid_loss.append(step_valid_loss)
        return np.mean(valid_loss)

    def valid_step(self, image, label):
        logit = self.model(image, training=False)
        iou_loss, conf_loss, prob_loss = self.loss_fn(label, logit)
        return iou_loss + conf_loss + prob_loss

    def export_model(self):
        tf.saved_model.save(self.model, self.params["saved_model_dir"])
        print("pb model saved in {}".format(self.params["saved_model_dir"]))


if __name__ == "__main__":
    # 💀 disable all gpus
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.set_visible_devices([], 'GPU')
    # ==========================================================================
    #                             make trainer instance                                  
    # ==========================================================================
    
    trainer = Trainer(params)
    
    # ==========================================================================
    #                             create dataloaders🛠️                                  
    # ==========================================================================
    
    DataReader = DataReader(
                                params["train_annotations_dir"],
                                img_size=params["img_size"],
                                transforms=transforms,
                                mosaic=params["mosaic_data"],
                                augment=params["augment_data"],
                                filter_idx=None,
                            )

    data_loader = DataLoader(
                                DataReader,
                                trainer.anchors,
                                trainer.stride,
                                params["img_size"],
                                params["anchor_assign_method"],
                                params["anchor_positive_augment"],
                            )
    
    
    train_dataset = data_loader(batch_size=params["batch_size"], anchor_label=True)
    train_dataset.len = len(DataReader)

    # for step, (image, target) in enumerate(train_dataset):
        
    #     print(f"target length {len(target)}")
    #     print(f"==========================================")
    #     print(f"target[0] length {target[0].shape}")
    #     print(f"target[1] length {target[1].shape}")
    #     print(f"target[2] length {target[2].shape}")
    #     print(f"==========================================")
    #     break


    #🔥 run actual training
    trainer.train(train_dataset=train_dataset, valid_dataset=train_dataset , transfer="resume")
    # trainer.train(train_dataset=train_dataset, valid_dataset=train_dataset , transfer="scratch")