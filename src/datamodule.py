from .Dataset import HierarchicalSublabelsObjectsDataset, FlatObjectsDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
import albumentations as A
import numpy as np




class MitosisTwoSubclassDataModule(pl.LightningDataModule):
    def __init__(self,
                img_dir: str,
                annotations_dict,
                num_train_samples:int =  500,
                num_val_samples: int = 200,
                crop_size = (256,256),
                labels = np.arange(8),
                batch_size = 8,
                num_workers = 8,
                train_transforms = None):
        
        super().__init__()
        
        self.crop_size = crop_size
        self.img_dir = img_dir
        self.annotations_dict = annotations_dict
        self.num_workers = num_workers
        self.labels = labels
        self.batch_size = batch_size

        # check for transformations passed
        if train_transforms == None:
            self.train_transform = None
        else:
            self.train_transform = train_transforms

        self.slide_names_train, self.slide_names_val = self.__create_train_val_split(annotations_dict,0.2)

        self.train_dataset = self.__create_dataset(annotations_dict, self.slide_names_train, img_dir, crop_size, num_train_samples,self.train_transform)
        self.val_dataset = self.__create_dataset(annotations_dict, self.slide_names_val, img_dir, crop_size, num_val_samples,None)


    def __create_dataset(self,
                        annotations_dict,
                        slide_names,
                        img_dir,
                        crop_size,
                        pseudo_epoch_length,
                        transformations):

        return HierarchicalSublabelsObjectsDataset(annotations_dict=annotations_dict,
                                      slide_names=slide_names,
                                      path_to_slides=img_dir,
                                      labels=self.labels,
                                      crop_size=crop_size,
                                      pseudo_epoch_length=pseudo_epoch_length,
                                      transformations=transformations)


    def __create_train_val_split(self,annotations_dict,val_split_size):
        train_imgs, val_imgs = train_test_split(list(annotations_dict.keys()),train_size = 1-val_split_size,test_size = val_split_size, random_state = 42)
        return train_imgs, val_imgs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers, collate_fn = self.train_dataset.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.val_dataset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers, collate_fn = self.val_dataset.collate_fn)
    

class MitosisDataModule(pl.LightningDataModule):
    def __init__(self,
                img_dir: str,
                annotations_dict,
                num_train_samples:int =  500,
                num_val_samples: int = 200,
                crop_size = (256,256),
                labels = np.arange(8),
                batch_size = 8,
                num_workers = 8,
                train_transforms = None):
        
        super().__init__()
        
        self.crop_size = crop_size
        self.img_dir = img_dir
        self.annotations_dict = annotations_dict
        self.num_workers = num_workers
        self.labels = labels
        self.batch_size = batch_size

        # check for transformations passed
        if train_transforms == None:
            self.train_transform = None
        else:
            self.train_transform = train_transforms

        self.slide_names_train, self.slide_names_val = self.__create_train_val_split(annotations_dict,0.2)

        self.train_dataset = self.__create_dataset(annotations_dict, self.slide_names_train, img_dir, crop_size, num_train_samples,self.train_transform)
        self.val_dataset = self.__create_dataset(annotations_dict, self.slide_names_val, img_dir, crop_size, num_val_samples,None)


    def __create_dataset(self,
                        annotations_dict,
                        slide_names,
                        img_dir,
                        crop_size,
                        pseudo_epoch_length,
                        transformations):

        return FlatObjectsDataset(annotations_dict=annotations_dict,
                                      labels=self.labels,
                                      slide_names=slide_names,
                                      path_to_slides=img_dir,
                                      crop_size=crop_size,
                                      pseudo_epoch_length=pseudo_epoch_length,
                                      transformations=transformations)


    def __create_train_val_split(self,annotations_dict,val_split_size):
        train_imgs, val_imgs = train_test_split(list(annotations_dict.keys()),train_size = 1-val_split_size,test_size = val_split_size, random_state = 42)
        return train_imgs, val_imgs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers, collate_fn = self.train_dataset.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.val_dataset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers, collate_fn = self.val_dataset.collate_fn)
