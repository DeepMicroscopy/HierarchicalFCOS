from typing import Dict, List, Tuple, Union
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
import pickle 
import numpy as np

#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, ChainedScheduler, OneCycleLR
from .model import make_two_level_hierarchical_fcos_model,make_fcos_model
from src.utils import match_scores_targets
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay






class LitFCOS(pl.LightningModule):
    def __init__(self,
                batch_size: int = 8,
                lr: float = 0.0001,
                num_classes: int = 11, 
                gamma: float = 0.1,
                optimizer: str = 'AdamW',
                scheduler: Union[str, None] = None,
                warmup_epochs: int = 5,	
                max_epochs: int = 100,
                patience: int = 10,
                pretrained_backbone: bool = True, 
                trainable_backbone_layers: int = 5, 
                means: List[float] = None,
                stds: List[float] = None,
                center_sampling_radius:float = 1.5,
                score_thresh:float = 0.2,
                detections_per_img: int = 100,
                topk_candidates: int = 1000,
                backbone:str = 'resnet50',
                min_size:int = 800,
                **kwargs) -> None:

        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.gamma = gamma
        self.means = means
        self.stds = stds
        self.automatic_optimization = False if self.scheduler == 'BaselineLRScheduler' else True
        self.patience = patience

        self.model = make_fcos_model(
            pretrained=pretrained_backbone,
            num_classes=num_classes,
            image_mean=means,
            image_std=stds,
            trainable_backbone_layers=trainable_backbone_layers,
            center_sampling_radius=center_sampling_radius,
            score_thresh=score_thresh,
            detections_per_img=detections_per_img,
            topk_candidates=topk_candidates,
            backbone=backbone,
            min_size=min_size,
            max_size=min_size,
            **kwargs)

        self.metric = MeanAveragePrecision(num_classes=self.num_classes, 
                                        iou_thresholds=[0.5], 
                                        class_metrics=True)
        

    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor = None) -> List[Dict[str, torch.Tensor]]:
        """Forward pass during inference.

        Returns post-processed predictions as a list of dictionaries.
        """
        return self.model(x, y)


    def training_step(self, batch, batch_idx):
        """Training step."""

        # forward pass
        images, targets = batch
        loss_dict = self(images, targets)
        # loss = sum(loss for loss in loss_dict.values())

        # loggging
        loss_cls = loss_dict['classification']
        loss_box = loss_dict['bbox_regression']
        loss_ctr = loss_dict['bbox_ctrness']

        # compute loss 
        loss = sum([loss_cls, loss_box, loss_ctr])

        # optimization
        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        # logging
        self.log('train/loss', loss, on_epoch=True, batch_size=self.batch_size)
        self.log('train/classification', loss_cls, on_epoch=True, batch_size=self.batch_size)
        self.log('train/bbox_regression', loss_box, on_epoch=True, batch_size=self.batch_size)
        self.log('train/bbox_centerness', loss_ctr, on_epoch=True, batch_size=self.batch_size)


        return loss 

    def validation_step(self, batch, batch_idx):
        """Validation step."""

        # forard pass 
        images, targets = batch
        preds, loss_dict = self(images, targets)
        # loss = sum(loss for loss in loss_dict.values())

        # loggging
        loss_cls = loss_dict['classification']
        loss_box = loss_dict['bbox_regression']
        loss_ctr = loss_dict['bbox_ctrness']
        # mse between targets and scores

        # compute loss 
        loss = sum([loss_cls, loss_box, loss_ctr])

        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=self.batch_size)
        self.log('val/loss_classifier', loss_cls, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('val/loss_box_reg', loss_box, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('val/loss_centerness', loss_ctr, on_epoch=True, on_step=False, batch_size=self.batch_size)

        # update metrics
        for idx, sample in enumerate(targets):
            if sample['boxes'].size(0) == 0:
                targets[idx]['boxes'] = torch.zeros((1, 4), dtype=torch.float32).to(self.device)
                # === geändert wegen dataset Änderung === #
                targets[idx]['labels'] = torch.tensor([0],dtype=torch.int64).to(self.device)

        self.metric.update(preds, targets)


    def validation_epoch_end(self, outputs):
        """Compute metrics at end of epoch."""
        metrics = self.metric.compute()
        ap_per_class = metrics['map_per_class']
        ar_per_class = metrics['mar_100_per_class']
        ap = ap_per_class if torch.numel(ap_per_class) < 2 else ap_per_class[1]
        ar = ar_per_class if  torch.numel(ar_per_class) < 2 else ar_per_class[1]

        self.log('val/ap', ap, prog_bar=True)
        self.log('val/ar', ar, prog_bar=True)
        self.metric.reset()



    def training_epoch_end(self, outputs):
        """Update the learning rate scheduler."""
        if not self.automatic_optimization:
            s1, s2 = self.lr_schedulers()
            s1.step()
            s2.step(self.trainer.callback_metrics['val/ap'])


    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(trainable_parameters, lr=self.lr)
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.lr)
        elif self.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(trainable_parameters, lr=self.lr)
        else:
            raise NotImplementedError

        if self.scheduler == 'BaselineLRScheduler':
            warmup_scheduler = {
                'scheduler': LinearLR(
                    optimizer, 
                    start_factor=1e-4, 
                    end_factor=1., 
                    total_iters=self.warmup_epochs,
                    verbose=False
                    ),
                'name': 'warmup_learning_rate',
                'interval': 'êpoch',
                'frequency': 1
                }
            lr_scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, 
                mode='max',
                factor=0.5,
                patience=self.patience,
                verbose=True,
                min_lr=1e-6
                ),
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1
            }
            return [optimizer], [warmup_scheduler, lr_scheduler]	

        elif self.scheduler == 'OneCycleLR':
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=self.lr, 
                total_steps=self.trainer.estimated_stepping_batches, 
                pct_start=0.3, 
                div_factor=10)
            return [optimizer], [scheduler]

        elif self.scheduler == None:
            return [optimizer]
        else:
            raise NotImplementedError


class SubClassAccuracyMetric():
    def __init__(self,num_classes, iou_threshold, radius):
        self.iou_threshold = iou_threshold
        self.radius = radius
        self.classes=num_classes
        self.reset()
    
    def reset(self):
        self.all_gt_scores = []
        self.all_predictions = []

    
    def update(self,preds,targets):
        for p,t in zip(preds,targets):
            boxes = p['boxes'].cpu().numpy()
            scores = p['scores'].cpu().numpy()
            scl_pred = p['sub_labels'].cpu().numpy()
            
            annos = t['boxes'].cpu().numpy()
            scl_targets = t['sub_labels'].cpu().numpy()
            gt_scores, pred_scores,_,_ = match_scores_targets(boxes,scores,scl_pred,annos,scl_targets, radius = self.radius, det_th = self.iou_threshold)
        self.all_gt_scores.extend(gt_scores)
        self.all_predictions.extend(pred_scores)
    
    def compute(self) -> float:
        if (len(self.all_gt_scores)>0):
            cm = confusion_matrix(y_true=self.all_gt_scores, y_pred=self.all_predictions)
            accuracy = np.sum((np.eye(cm.shape[0])*cm))/np.sum(cm[1:,:])
        else:
            accuracy = 0.
        stats = {'y_true': self.all_gt_scores, 'y_pred': self.all_predictions}

        return accuracy, stats

class SubClassAccuracyMetricL2():
    def __init__(self,num_classes, iou_threshold, radius):
        self.iou_threshold = iou_threshold
        self.radius = radius
        self.classes=num_classes
        self.reset()
    
    def reset(self):
        self.all_gt_scores = []
        self.all_predictions = []
        self.all_gt_scores_l2 = []
        self.all_predictions_l2 = []

    
    def update(self,preds,targets):
        for p,t in zip(preds,targets):
            boxes = p['boxes'].cpu().numpy()
            scores = p['scores'].cpu().numpy()
            scl_pred = p['sub_labels'].cpu().numpy()
            scl_pred_l2 = p['sub_labels_l2'].cpu().numpy()
            
            annos = t['boxes'].cpu().numpy()
            scl_targets = t['sub_labels'].cpu().numpy()
            scl_targets_l2 = t['sub_labels_l2'].cpu().numpy()
            gt_scores, pred_scores,_,_ = match_scores_targets(boxes,scores,scl_pred,annos,scl_targets, radius = self.radius, det_th = self.iou_threshold)
            gt_scores_l2, pred_scores_l2,_,_ = match_scores_targets(boxes,scores,scl_pred_l2,annos,scl_targets_l2, radius = self.radius, det_th = self.iou_threshold)
        self.all_gt_scores.extend(gt_scores)
        self.all_predictions.extend(pred_scores)
        self.all_gt_scores_l2.extend(gt_scores_l2)
        self.all_predictions_l2.extend(pred_scores_l2)
    
    def compute(self) -> float:
        if (len(self.all_gt_scores)>0):
            cm = confusion_matrix(y_true=self.all_gt_scores, y_pred=self.all_predictions)
            accuracy = np.sum((np.eye(cm.shape[0])*cm))/np.sum(cm[1:,:])
        else:
            accuracy = 0.
        if (len(self.all_gt_scores_l2)>0):
            cm = confusion_matrix(y_true=self.all_gt_scores_l2, y_pred=self.all_predictions_l2)
            accuracy_l2 = np.sum((np.eye(cm.shape[0])*cm))/np.sum(cm[1:,:])
        else:
            accuracy_l2 = 0.
        stats = {'y_true': self.all_gt_scores, 'y_pred': self.all_predictions,
                 'y_true_l2': self.all_gt_scores_l2, 'y_pred_l2': self.all_predictions_l2}

        return accuracy, accuracy_l2, stats    
    

class LitHFCOS(pl.LightningModule):
    def __init__(self,
                 subclass_names:str,
                batch_size: int = 8,
                lr: float = 0.0001,
                num_classes: int = 2, 
                num_subclasses : int = 2,
                num_subclasses_l2 : int = 4,
                gamma: float = 0.1,
                optimizer: str = 'AdamW',
                scheduler: Union[str, None] = None,
                warmup_epochs: int = 5,	
                max_epochs: int = 100,
                patience: int = 10,
                pretrained_backbone: bool = True, 
                trainable_backbone_layers: int = 5, 
                means: List[float] = None,
                stds: List[float] = None,
                center_sampling_radius:float = 1.5,
                score_thresh:float = 0.2,
                detections_per_img: int = 100,
                topk_candidates: int = 1000,
                backbone:str = 'resnet50',
                min_size:int = 800,
                model = None,
                **kwargs) -> None:

        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.gamma = gamma
        self.means = means
        self.subclass_w = [1.0, 1.0]
        self.stds = stds
        self.subclass_names = subclass_names
        self.automatic_optimization = False if self.scheduler == 'BaselineLRScheduler' else True
        self.patience = patience
        if model is None:
            self.model = make_two_level_hierarchical_fcos_model(
                pretrained=pretrained_backbone,
                num_classes=num_classes,
                num_sub_classes_level2=num_subclasses_l2,
                num_sub_classes = num_subclasses,
                image_mean=means,
                image_std=stds,
                trainable_backbone_layers=trainable_backbone_layers,
                center_sampling_radius=center_sampling_radius,
                score_thresh=score_thresh,
                detections_per_img=detections_per_img,
                topk_candidates=topk_candidates,
                backbone=backbone,
                min_size=min_size,
                max_size=min_size,
                **kwargs)
        else:
            self.model = model

        self.subclass_accuracy_metric = SubClassAccuracyMetricL2(num_classes=9, iou_threshold=0.5, radius=25)
        self.metric = MeanAveragePrecision(num_classes=self.num_classes, 
                                        iou_thresholds=[0.5], 
                                        class_metrics=True)
        

    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor = None) -> List[Dict[str, torch.Tensor]]:
        """Forward pass during inference.

        Returns post-processed predictions as a list of dictionaries.
        """
        return self.model(x, y)


    def training_step(self, batch, batch_idx):
        """Training step."""

        # forward pass
        images, targets = batch
        loss_dict = self(images, targets)
        # loss = sum(loss for loss in loss_dict.values())

        # loggging
        loss_cls = loss_dict['classification']
        loss_box = loss_dict['bbox_regression']
        loss_ctr = loss_dict['bbox_ctrness']
        loss_sbc = loss_dict['sub_cls']
        loss_sbc_l2 = loss_dict['sub_cls_l2']

        # compute loss 
        loss = sum([loss_cls, loss_box, loss_ctr, self.subclass_w[0]*loss_sbc, self.subclass_w[1]*loss_sbc_l2])

        # optimization
        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        # logging
        self.log('train/loss', loss, on_epoch=True, batch_size=self.batch_size)
        self.log('train/classification', loss_cls, on_epoch=True, batch_size=self.batch_size)
        self.log('train/bbox_regression', loss_box, on_epoch=True, batch_size=self.batch_size)
        self.log('train/bbox_centerness', loss_ctr, on_epoch=True, batch_size=self.batch_size)
        self.log('train/sub_cls', loss_sbc, on_epoch=True, batch_size=self.batch_size)


        return loss 

    def validation_step(self, batch, batch_idx):
        """Validation step."""

        # forard pass 
        images, targets = batch
        preds, loss_dict = self(images, targets)
        # loss = sum(loss for loss in loss_dict.values())

        # loggging
        loss_cls = loss_dict['classification']
        loss_box = loss_dict['bbox_regression']
        loss_ctr = loss_dict['bbox_ctrness']
        loss_sbc = loss_dict['sub_cls']
        loss_sbc_l2 = loss_dict['sub_cls_l2']

        # compute loss 
        loss = sum([loss_cls, loss_box, loss_ctr, loss_sbc, loss_sbc_l2])

        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=self.batch_size)
        self.log('val/loss_classifier', loss_cls, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('val/loss_box_reg', loss_box, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('val/loss_centerness', loss_ctr, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('val/loss_sbc', loss_sbc, on_epoch=True, batch_size=self.batch_size)
        self.log('val/loss_sbc_l2', loss_sbc_l2, on_epoch=True, batch_size=self.batch_size)

        # update metrics
        for idx, sample in enumerate(targets):
            if sample['boxes'].size(0) == 0:
                targets[idx]['boxes'] = torch.zeros((1, 4), dtype=torch.float32).to(self.device)
                # === geändert wegen dataset Änderung === #
                targets[idx]['labels'] = torch.tensor([0],dtype=torch.int64).to(self.device)

        self.metric.update(preds, targets)
        self.subclass_accuracy_metric.update(preds,targets)


    def validation_epoch_end(self, outputs):
        """Compute metrics at end of epoch."""
        metrics = self.metric.compute()
        ap_per_class = metrics['map_per_class']
        ar_per_class = metrics['mar_100_per_class']
        
        acc,acc_l2,stats = self.subclass_accuracy_metric.compute()
        
        ap = ap_per_class if torch.numel(ap_per_class) < 2 else ap_per_class[1]
        ar = ar_per_class if  torch.numel(ar_per_class) < 2 else ar_per_class[1]

        self.log('val/ap', ap, prog_bar=True)
        self.log('val/ar', ar, prog_bar=True)
        
        self.log('val/subclass_acc', acc, prog_bar=True)
        self.log('val/subclass_acc_l2', acc_l2, prog_bar=True)
        if len(stats['y_pred'])>0:
            # calculate and log confusion matrix. +1 is required to compensate for "-1" target value
            wandb.log({"val/subclass_cm": wandb.plot.confusion_matrix(probs=None,
                            y_true=[x+1 for x in stats['y_true']], preds=[x + 1 for x in stats['y_pred']],
                            class_names=self.subclass_names)})
        
        self.metric.reset()
        self.subclass_accuracy_metric.reset()



    def training_epoch_end(self, outputs):
        """Update the learning rate scheduler."""
        
        if not self.automatic_optimization:
            s1, s2 = self.lr_schedulers()
            s1.step()
            s2.step(self.trainer.callback_metrics['val/ap'])


    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(trainable_parameters, lr=self.lr)
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.lr)
        elif self.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(trainable_parameters, lr=self.lr)
        else:
            raise NotImplementedError

        if self.scheduler == 'BaselineLRScheduler':
            warmup_scheduler = {
                'scheduler': LinearLR(
                    optimizer, 
                    start_factor=1e-4, 
                    end_factor=1., 
                    total_iters=self.warmup_epochs,
                    verbose=False
                    ),
                'name': 'warmup_learning_rate',
                'interval': 'êpoch',
                'frequency': 1
                }
            lr_scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, 
                mode='max',
                factor=0.5,
                patience=self.patience,
                verbose=True,
                min_lr=1e-6
                ),
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1
            }
            return [optimizer], [warmup_scheduler, lr_scheduler]	

        elif self.scheduler == 'OneCycleLR':
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=self.lr, 
                total_steps=self.trainer.estimated_stepping_batches, 
                pct_start=0.3, 
                div_factor=10)
            return [optimizer], [scheduler]

        elif self.scheduler == None:
            return [optimizer]
        else:
            raise NotImplementedError
