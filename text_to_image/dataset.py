
import numpy as np
import torch.utils.data
import torchvision
from tqdm import tqdm

class NamsBaseDataset(torch.utils.data.Dataset):
    
    def name(self):
        assert(False)
        
    def get_image_path(self, idx):
        assert(False)
    
    def get_image_captions(self, idx):
        assert(False)
        
    def get_loader(self, batch_size, shuffle = False, drop_last = False, num_workers = 0):
        return torch.utils.data.DataLoader(
            self,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            drop_last = drop_last,
            collate_fn = lambda i: i
        )
    
    def normalize_caption(self, text):
        import string
        if type(text) == str:
            text = unicode(text, "utf-8")
        text = text.encode('ascii', 'replace')
        text = str(text).lower().translate(None, string.punctuation).strip()
        return text
    
    def get_all_texts(self):
        texts = []
        for i in range(len(self)):
            for t in self.get_image_captions(i):
                texts.append(t)
        return texts
    
    def precompute_img_features(self, force=False):
        features_filename = self.name() + '_features.npy'
        try:
            assert(not force)
            self.img_features = np.load(features_filename)
            print 'sucessfully loaded features'
            return
        except:
            print 'compute features...'
        self.img_features = None
        
        # run model on all images
        net = torchvision.models.resnet50(pretrained=True)
        net.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        net.fc = torch.nn.Dropout()
        net = net.cuda().eval()
        loader = self.get_loader(batch_size=8, shuffle=False, drop_last=False, num_workers=4)
        img_features = np.zeros((len(self), 2048))
        i = 0
        for data in tqdm(loader):
            imgs = torch.stack([d['image'] for d in data])
            x = net(imgs.cuda()).cpu().detach().numpy()
            img_features[i:(i+x.shape[0]),:] = x
            imgs = torch.flip(imgs, [3])
            x = net(imgs.cuda()).cpu().detach().numpy()
            img_features[i:(i+x.shape[0]),:] += x
            i += x.shape[0]
        self.img_features = img_features
        np.save(features_filename, self.img_features)
    
    def __getitem__(self, idx):
        if self.img_features is not None:
            img = self.img_features[idx,:]
        else:
            raw_img = torchvision.datasets.folder.pil_loader(self.get_image_path(idx))
            img = raw_img
            if self.transform:
                img = self.transform(img)
        img = {
            'id': None,
            'label': None,
            'index': idx,
            'image': img,
            'captions': self.get_image_captions(idx)
        }
        return img

# Required files:
# dataset_path/annotations/captions_val2014.json
# dataset_path/annotations/captions_train2014.json
# dataset_path/train2014/[image files]
# dataset_path/val2014/[image files]
class COCOCaptionDataset(NamsBaseDataset):
    
    def __init__(self, dataset_path = '', transform = None, test_split = False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.test_split = test_split
        
        import json
        if test_split:
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
            imgs[id2id[cap['image_id']]]['captions'] += [self.normalize_caption(cap['caption'])]

        self.imgs = imgs
    
    def __len__(self):
        return len(self.imgs)

    def name(self):
        if self.test_split:
            return "CocoCapTest"
        return "CocoCapTrain"
        
    def get_image_path(self, idx):
        return self.imgs[idx]['filename']
    
    def get_image_captions(self, idx):
        return self.imgs[idx]['captions']
    
    def __getitem__(self, idx):
        item = super(COCOCaptionDataset, self).__getitem__(idx)
        item['label'] = self.imgs[idx]['id']
        return item
        
class SimpleImageCaptions112(NamsBaseDataset):
    
    def __init__(self, dataset_path = '', transform = None):
        self.dataset_path = dataset_path
        self.transform = transform
        imgs = []
        import os
        import os.path
        for d in os.listdir(dataset_path):
          if not os.path.isfile(dataset_path + '/' + d):
            for f in os.listdir(dataset_path + '/' + d):
              if os.path.isfile(dataset_path + '/' + d + '/' + f):
                imgs += [{
                    'id': len(imgs),
                    'captions': [self.normalize_caption(d)],
                    'filename': dataset_path + '/' + d + '/' + f
                }]
        self.imgs = imgs
        self.make_test_queries()
    
    def __len__(self):
        return len(self.imgs)
    
    def name(self):
        return "SimpleImageCaptions112"
        
    def get_image_path(self, idx):
        return self.imgs[idx]['filename']
    
    def get_image_captions(self, idx):
        return self.imgs[idx]['captions']
    
    def __getitem__(self, idx):
        item = super(SimpleImageCaptions112, self).__getitem__(idx)
        item['label'] = self.imgs[idx]['captions'][0]
        return item
    
    def make_test_queries(self):
        
        novel_obj_list = ['trex', 'stormtrooper', 'darthvader', 'chewbacca']
        
        caption2ids = {}
        for i in range(len(self)):
            for caption in self.get_image_captions(i):
                try:
                    caption2ids[caption] += [i]
                except:
                    caption2ids[caption] = []
                    caption2ids[caption] += [i]
        
        test_queries = []
        for cap1 in caption2ids.keys():
            for cap2 in caption2ids.keys():
                cap1s = cap1.replace('on the ', '').replace('in the ', '').replace('living room', 'livingroom').split()
                cap2s = cap2.replace('on the ', '').replace('in the ', '').replace('living room', 'livingroom').split()
                diffs = []
                for w1, w2 in zip(cap1s, cap2s):
                    if w1 != w2:
                        w1 = w1.replace('livingroom', 'living room')
                        w2 = w2.replace('livingroom', 'living room')
                        diffs += [w1, w2]
                if len(diffs) != 2:
                    continue
                if diffs[0] in novel_obj_list or diffs[1] in novel_obj_list:
                    continue
                for idx in caption2ids[cap1]:
                    test_queries += [{
                        'source_idx': idx,
                        'source_caption': cap1,
                        'target_caption': cap2,
                        'replacing_words': diffs
                    }]
        self.test_queries_seen_objects = []
        self.test_queries_novel_objects = []
        for t in test_queries:
            novel_objects = False
            for w in novel_obj_list:
                if w in t['source_caption']:
                    novel_objects = True
            if novel_objects:
                self.test_queries_novel_objects += [t]
            else:
                self.test_queries_seen_objects += [t]
        print len(self.test_queries_seen_objects), len(self.test_queries_novel_objects)
        #assert(len(self.test_queries_seen_objects) == 18051)
        assert(len(self.test_queries_novel_objects) == 745)
 