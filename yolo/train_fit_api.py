from bentoml import Model
import tensorflow as tf
from rich.console import Console
from pathlib import Path
import os
from yolo import Yolo, params
# from .configs import params
import wandb
from yolo import DataLoader, DataReader, transforms, YoloLoss, LrScheduler, Optimizer, YoloLoss_fit_api
from wandb.keras import WandbCallback

# clear session 
# tf.keras.backend.clear_session()

class Trainer(object):
    
    def __init__(self, run, params) -> None:
        self.params = params
        self.run = run

        self._get_strategy()
        
        self._build_model()
        
        (self.train_dataset, self.train_data_len), (self.test_dataset, self.test_data_len) = self._get_dataloader()
        
        
    def _get_strategy(self,):
        self.strategy = tf.distribute.MirroredStrategy()
        # self.strategy =  tf.distribute.OneDeviceStrategy(device="/gpu:0")
        
        
    
    def _build_model(self):
        
        
            self.model_class = Yolo(yaml_dir=self.params["yaml_dir"])
            self.anchors = self.model_class.module_list[-1].anchors
            self.stride = self.model_class.module_list[-1].stride
            self.num_classes = self.model_class.module_list[-1].num_classes
            self.model = self.model_class(img_size=self.params["img_size"])
            # print model summary
            print(f"model summaruy is \n\n {self.model.summary()}")
            print(f"==============================================")
            print(f"anchors: {self.anchors}")
            print(f"stride: {self.stride}")
            print(f"num_classes: {self.num_classes}")
            print(f"==============================================")
            
    
    
    
    def _get_dataloader(self,):
        
            # define dataset-reader
            Reader = DataReader(
                                        self.params["train_annotations_dir"],
                                        img_size=self.params["img_size"],
                                        transforms=transforms,
                                        mosaic=self.params["mosaic_data"],
                                        augment=self.params["augment_data"],
                                        filter_idx=None,
                                    )
            # define dataset-iterator
            data_loader = DataLoader(
                                        Reader,
                                        self.anchors,
                                        self.stride,
                                        self.params["img_size"],
                                        self.params["anchor_assign_method"],
                                        self.params["anchor_positive_augment"],
                                    )
        
            
            train_dataset = data_loader(batch_size=self.params["batch_size"], anchor_label=True)
            test_dataset = data_loader(batch_size=self.params["batch_size"], anchor_label=True)
            train_data_len = len(Reader)
            test_data_len = len(Reader)
            
            return (train_dataset, train_data_len), (test_dataset, test_data_len)
                
    
    
    
    
    def _configure_optimizer(self,):
        self.optimizer = Optimizer("adam")()
    
    
    
    def _configure_loss(self,):
        '''
        get loss function for yolo, should return the total loss as scalar
        '''        
        self.loss_fn = YoloLoss_fit_api(
                                            self.anchors,
                                            ignore_iou_threshold=0.3,
                                            num_classes=self.num_classes,
                                            label_smoothing=self.params["label_smoothing"],
                                            img_size=self.params["img_size"],
                                        )


    def _get_callbacks(self,):
        
            wandb_callback = WandbCallback(
                            monitor="val_loss", verbose=0, mode="min", save_weights_only=False,
                            log_weights=True, log_gradients=(False), save_model=(True),
                            training_data=None, validation_data=None, labels=[], predictions=36,
                            generator=None, input_type=None, output_type=None, log_evaluation=(False),
                            validation_steps=None, class_colors=None, log_batch_frequency=None,
                            log_best_prefix="best_", save_graph=True, validation_indexes=None,
                            validation_row_processor=None, prediction_row_processor=None,
                            infer_missing_processors=(True), log_evaluation_frequency=0,
                            compute_flops=(True)
                        )
            # define model chackpoint callback
            ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.params["checkpoint_dir"],
                monitor="val_loss",
                verbose=0,
                save_best_only=False,
                save_weights_only=False,
                mode="min",
                save_freq="epoch",
                options=None,
            )
            # # define early stopping callback
            # EarlyStopping = tf.keras.callbacks.EarlyStopping(
            #     monitor="val_loss",
            #     min_delta=0,
            #     patience=10,
            #     verbose=0,
            #     mode="auto",
            #     baseline=None,
            #     restore_best_weights=True,
            # )
            # define learning rate scheduler callback
            # LearningRateScheduler = LrScheduler(self.params["lr_scheduler"])
            # define callbacks
            callbacks = [ModelCheckpoint, 
                        #  EarlyStopping, 
                        #  LearningRateScheduler, 
                        wandb_callback]
            
            self.callbacks = callbacks


    def train(self):
        
        
        # define the loss, optimizer and callbacks under strategy scope
        self._configure_optimizer()
        self._configure_loss()
        self._get_callbacks()

        self.model.compile(
                            optimizer=self.optimizer,
                            loss=self.loss_fn,
                            # metrics=["accuracy"],
                        )
        
        # ⚡ start training the model
        self.model.fit(
                        self.train_dataset,
                        epochs=self.params["n_epochs"],
                        steps_per_epoch=self.train_data_len // self.params["batch_size"],
                        validation_data=self.test_dataset,
                        validation_steps=self.test_data_len // self.params["batch_size"],
                        callbacks=self.callbacks,
                    )
        
        # 💾 save the model as tf.keras.model
        self.model.save(self.params["save_weights_dir"]+"/final_model")
        

    
    
if __name__ == "__main__":
    # 💀 disable all gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([], 'GPU')
    # 🔎 start training
    run = wandb.init(project="yolo-tf", entity="bb_ai-team")
    trainer = Trainer(run, params)
    trainer.train()
    
    
    run.finish()
    