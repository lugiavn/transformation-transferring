import numpy as np
import PIL
import skimage.io
import torch
import json
import torch.utils.data
import torchvision
import warnings
import random
from PIL import Image
from skimage import io
import json

class NamsBaseDataset(torch.utils.data.Dataset):
    
    def __len__(self):
        return len(self.imgs)
    
    def name(self):
        assert(False)

    def enable_skip_image_data(self, enable=True):
        self.skip_image_data = enable
        return self
    
    def normalize_captions(self):
        import string
        for img in self.imgs:
            caps = []
            for text in img['captions']:
                if type(text) == str:
                    text = unicode(text, "utf-8")
                text = text.encode('ascii', 'replace')
                text = str(text).lower().translate(None, string.punctuation).strip()
                caps.append(text)
            img['captions'] = caps
    
    def __getitem__(self, idx):
        if self.skip_image_data:
            raw_img = None
            img = None
        else:
            if False:
                raw_img = io.imread(self.imgs[idx]['filename'])
                raw_img = Image.fromarray(raw_img)
            else:
                raw_img = torchvision.datasets.folder.pil_loader(self.imgs[idx]['filename'])
            img = raw_img
            if self.transform:
                img = self.transform(img)
        img = {
            'id': self.imgs[idx]['id'],
            'raw_image': raw_img,
            'image': img,
            'captions': self.imgs[idx]['captions']
        }
        return img
        
    def get_loader(self, batch_size, shuffle = False, drop_last = False, num_workers = 0):
        return torch.utils.data.DataLoader(
            self,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            drop_last = drop_last,
            collate_fn = lambda i: i
        )
    
    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img['captions']:
                texts.append(c)
        return texts
    
class COCOCaptionDataset(NamsBaseDataset):
    
    def __init__(self, dataset_path = '', transform = None, valset = False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.valset = valset
        
        if valset:
            x = json.load(open(self.dataset_path + '/annotations/captions_val2014.json', 'rt'))
            img_path = dataset_path + '/val2014/'
        else:
            x = json.load(open(self.dataset_path + '/annotations/captions_train2014.json', 'rt'))
            img_path = dataset_path + '/train2014/'
            
        imgs = []
        id2id = {}
        for img in x['images']:
            id2id[img['id']] = len(imgs)
            imgs += [{
                'id': img['id'],
                'class': img['id'],
                'filename': img_path + img['file_name'],
                'captions': []
            }]

        for cap in x['annotations']:
            imgs[id2id[cap['image_id']]]['captions'] += [cap['caption']]

        self.imgs = imgs

    def name(self):
        if self.valset:
            return "CocoCap2014val"
        else:
            return "CocoCap2014train"

class SIC112Dataset(NamsBaseDataset):
    
    def __init__(self, dataset_path = '', transform = None):
        pass

    def name(self):
        return "SIC112"
