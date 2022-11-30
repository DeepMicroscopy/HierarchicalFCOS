import torch
import os
import numpy as np
import platform

from PIL import Image
from torchvision import transforms
from numpy import random
from torchvision.transforms.functional import to_tensor        

class MultiClassObjectsDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_dict, slide_names, path_to_slides, labels, crop_size = (128,128), pseudo_epoch_length:int = 1000, transformations = None, sampling='randomCategories'):
        super().__init__()
        
        self.separator = os.sep

        self.anno_dict = annotations_dict
        self.slide_names = slide_names
        self.path_to_slides = path_to_slides
        self.crop_size = crop_size
        self.pseudo_epoch_length = pseudo_epoch_length
        # highest value in labels
        self.labels = labels
         
        # list which holds annotations of all slides in slide_names in the format
        # slide_name, annotation, label, min_x, max_x, min_y, max_y
        
        self.slide_dict, self.annotations_list, self.labels_ordered_dict = self._initialize()
        if (sampling=='randomCategories'):
            self.sample_cord_list = self._sample_cord_list()
        else:
            self.sample_cord_list = self._continuous_cord_list()
            
        self.transformations = transformations 
        # set up transformations
        if transformations != None:
            self.transform = transformations
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])


    def _initialize(self):
        # open all images and store them in self.slide_dict with their name as key value
        slide_dict = {}
        
        labels_ordered_dict = {l : [] for l in self.labels}
        annotations_list = []
        for slide in self.slide_names:
            if os.path.exists(slide):
                im_obj = Image.open(self.path_to_slides + self.separator + slide).convert('RGB')
                slide_dict[slide] = im_obj
                # setting up a list with all bounding boxes
                for annotation in self.anno_dict[slide]:
                    max_x, min_x = max(self.anno_dict[slide][annotation]['x']), min(self.anno_dict[slide][annotation]['x'])
                    max_y, min_y = max(self.anno_dict[slide][annotation]['y']), min(self.anno_dict[slide][annotation]['y'])
                    label = self.anno_dict[slide][annotation]['class']
                    
                    labels_ordered_dict[label].append([slide, annotation, label, min_x, min_y, max_x, max_y])
                    annotations_list.append([slide, annotation, label, min_x, min_y, max_x, max_y])


        return slide_dict, annotations_list, labels_ordered_dict


        
    def _continuous_cord_list(self):
        slides = np.zeros(0)
        centers = np.zeros((0,2), np.int64)
        for slide in self.slide_names:
            width,height = self.slide_dict[slide].size
            for x in torch.arange(int(self.crop_size[0]/2),width,self.crop_size[0]-64):
                for y in torch.arange(int(self.crop_size[1]/2),height, self.crop_size[1]-64):
                    centers= np.concatenate((centers,np.array([[int(x),int(y)]])))
                    slides = np.append(slides,slide)
        return np.concatenate((slides.reshape(-1,1),centers), axis = -1)

    
    def _sample_cord_list(self, offset = 50):
        # select random labels
        labels = random.choice(self.labels, size = self.pseudo_epoch_length, replace = True)
        # select coordinates where these labels are present
        print(f'{[len(i) for i in self.labels_ordered_dict.values()]}')
        
        indice = [random.choice(len(self.labels_ordered_dict[l])) for l in labels]
        annos = [self.labels_ordered_dict[l][i] for l,i in zip(labels,indice)]
        
        slides = np.array([i[0] for i in annos])
        centers = np.array([((i[3] + i[5]) //2, (i[4] + i[6]) //2) for i in annos])
        # center crop around cell
        centers[:,0] = centers[:,0] - (self.crop_size[0] // 2)
        centers[:,1] = centers[:,1] - (self.crop_size[1] // 2)
        # sample random offsets
        # select coordinates from which to load images
        # only works if all images have the same size
        width,height = self.slide_dict[slides[0]].size

        offsets = np.random.randint(low = 0, high = offset, size = (len(centers),2))
        centers = centers + offsets
        # check wether centers still in image
        centers = self.__check_centers(centers,width,height)
        
        return np.concatenate((slides.reshape(-1,1),centers), axis = -1)
        
    def __check_centers(self,centers,width,height):
        
        for i,(x,y) in enumerate(centers):
            if x < 0:
                centers[i][0] = 0
            elif x > width - self.crop_size[0]:
                centers[i][0] = width - self.crop_size[0]
            if y < 0:
                centers[i][1] = 0
            elif y > height - self.crop_size[1]:
                centers[i][1] = height - self.crop_size[1]
            
        return centers

    def __len__(self):
        return len(self.sample_cord_list)

    def _get_boxes_and_label(self,slide,x_cord,y_cord):
        return [line[2::] for line in self.annotations_list if line[0] == slide and line[3] > x_cord and line [4] > y_cord and line[5] < x_cord + self.crop_size[0] and line[6] < y_cord + self.crop_size[1]]
    
    def _check_for_degenerated_boxes(self,boxes,labels, th = 5):
        '''
        checks for boxes which are to small due to sampling
        '''
        
        idx = [row for row, box in enumerate(boxes) if box[2] - box[0] >= th or box[3] - box[1] >= th]
        return boxes[idx], labels[idx]
                
        
        

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __iter__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        targets = list()


        for b in batch:
            images.append(b[0])
            targets.append(b[1])
            
            
        images = torch.stack(images, dim=0)

        return images, targets

    def trigger_sampling(self):
        self.sample_cord_list = self._sample_cord_list()
        
        
class FlatObjectsDataset(MultiClassObjectsDataset):


    def __getitem__(self,index):
        slide, x_cord, y_cord = self.sample_cord_list[index]
        x_cord = np.int64(x_cord)
        y_cord = np.int64(y_cord)
        # load image
        img = self.slide_dict[slide].crop((x_cord,y_cord,x_cord + self.crop_size[0],y_cord + self.crop_size[1]))
        
        # load boxes for the image
        labels_boxes = self._get_boxes_and_label(slide,x_cord,y_cord)
        # check if there is no labeld instance on the image
        if len(labels_boxes) == 0:
            boxes = torch.zeros((0,4),dtype = torch.float32)
            labels = torch.tensor([0], dtype = torch.int64)
        else:
            boxes = torch.tensor([[line[1] - x_cord, line[2] - y_cord, line[3] - x_cord, line[4] - y_cord] for line in labels_boxes],dtype=torch.float32)

            
        # check if boxes are valid
        labels = torch.tensor([line[0] for line in labels_boxes], dtype=torch.int64)
        boxes, labels = self._check_for_degenerated_boxes(boxes,labels)
        #labels = torch.ones_like(sub_labels, dtype = torch.int64)
        
        # applay transformations
        if self.transformations != None and len(boxes) > 0:
            transformed = self.transform(image = np.array(img),
                                        bboxes = boxes,
                                        labels = labels)
            
            boxes = torch.tensor(transformed['bboxes'], dtype = torch.float32)
            labels = torch.tensor(transformed['labels'], dtype = torch.long)
            img = to_tensor(transformed['image'])
            
        else:
            img = to_tensor(img)
        
        
        
        
        # check if boxes are valid
        boxes, labels = self._check_for_degenerated_boxes(boxes,labels)
        # check if boxes left, if not insert empty one
        if len(boxes) == 0:
            boxes = torch.zeros((0,4),dtype = torch.float32)
            labels = torch.tensor([0], dtype = torch.int64)

        target = {
            'boxes': boxes,
            'labels':labels,
            'slide':slide
        }

        return img, target
        
    
class HierarchicalSublabelsObjectsDataset(MultiClassObjectsDataset):


    def __getitem__(self,index):
        slide, x_cord, y_cord = self.sample_cord_list[index]
        x_cord = np.int64(x_cord)
        y_cord = np.int64(y_cord)
        # load image
        img = self.slide_dict[slide].crop((x_cord,y_cord,x_cord + self.crop_size[0],y_cord + self.crop_size[1]))
        
        # load boxes for the image
        labels_boxes = self._get_boxes_and_label(slide,x_cord,y_cord)
        
        # check if there is no labeld instance on the image
        if len(labels_boxes) == 0:
            sub_labels = torch.tensor([-1], dtype = torch.int64)
            boxes = torch.zeros((0,4),dtype = torch.float32)
            labels = torch.tensor([0], dtype = torch.int64)
        else:
            sub_labels = torch.tensor([line[0] for line in labels_boxes], dtype=torch.int64)
            # now, you need to change the originale box cordinates to the cordinates of the image
            boxes = torch.tensor([[line[1] - x_cord, line[2] - y_cord, line[3] - x_cord, line[4] - y_cord] for line in labels_boxes],dtype=torch.float32)

            
        # check if boxes are valid
        boxes, sub_labels = self._check_for_degenerated_boxes(boxes,sub_labels)
        labels = torch.ones_like(sub_labels, dtype = torch.int64)
        
        # applay transformations
        if self.transformations != None and len(boxes) > 0:
            transformed = self.transform(image = np.array(img),
                                        bboxes = boxes,
                                        labels = labels,
                                        sub_labels = sub_labels)
            
            boxes = torch.tensor(transformed['bboxes'], dtype = torch.float32)
            labels = torch.tensor(transformed['labels'], dtype = torch.long)
            sub_labels = torch.tensor(transformed['sub_labels'], dtype = torch.int64)
            img = to_tensor(transformed['image'])
            
        else:
            img = to_tensor(img)
        
        
        
        
        # check if boxes are valid
        boxes, sub_labels = self._check_for_degenerated_boxes(boxes,sub_labels)
        # check if boxes left, if not insert empty one
        if len(boxes) == 0:
            sub_labels = torch.tensor([-1], dtype = torch.int64)
            boxes = torch.zeros((0,4),dtype = torch.float32)
            labels = torch.tensor([0], dtype = torch.int64)
        else:
            # construct label tensor
            labels = torch.ones_like(sub_labels, dtype = torch.int64)

        # label is 1 for cell and 0 for background
        target = {
            'boxes': boxes,
            'labels':labels, # 1 for mitosis
            'sub_labels':(sub_labels>0).long()-(sub_labels<0).long(), # -1 for ambiguous, 0 for AMF or 1 for typical MF
            'sub_labels_l2': sub_labels-1+(sub_labels<0).long(), # 0 to 3 for typical subtypes, else -1
            'slide':slide
        }

        return img, target    