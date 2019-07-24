# Nam
import time
import numpy as np
import random
from tqdm import tqdm
import sys
import argparse
from tensorboardX import SummaryWriter
import torch
import torchvision

from model import *
from dataset import *
from test_retrieval import *
from word_replacing import *

torch.set_num_threads(3)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='f')
    parser.add_argument('--comment', type=str, default='x')
    parser.add_argument('--coco_path', type=str, default='./coco')
    parser.add_argument('--sic112_path', type=str, default='./downloads12')
    parser.add_argument('--transformation_architecture', type=str, default='modified_tirg')
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--loader_num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=320)
    parser.add_argument('--optim', type=str, default='adam', help='what update to use? sgd|adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_decay_frequency', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=1e-05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_epochs', type=int, default=50)
    args = parser.parse_args()
    return args

def load_datasets(opt):
    transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(336, scale=(0.8, 1.0), ratio=(0.75, 1.3)),
            #torchvision.transforms.Resize((336,336)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trainset = COCOCaptionDataset(
        opt.coco_path,
        transform = transform,
        test_split = False
    )
    testset = COCOCaptionDataset(
        opt.coco_path,
        transform = transform,
        test_split = True
    )
    sic112 = SimpleImageCaptions112(
        opt.sic112_path,
        transform = transform
    )
    trainset.precompute_img_features(0)
    testset.precompute_img_features(0)
    sic112.precompute_img_features(0)
    return trainset, testset, sic112

def create_model(opt, trainset):
    model = ImageTextEncodeTransformModel(opt.embed_dim, trainset.get_all_texts())
    if opt.transformation_architecture == 'modified_tirg':
        model.transformer = MTirgTransform(opt.embed_dim)
    elif opt.transformation_architecture == 'concat':
        model.transformer = ConcatTransform(opt.embed_dim)
    elif opt.transformation_architecture == 'tirg':
        model.transformer = TirgTransform(opt.embed_dim)
    model = model.cuda()
    
    # create optimizer
    params = []
    params.append({'params': [p for p in model.img_encoder.fc.parameters()]})
    params.append({'params': [p for p in model.img_encoder.parameters()], 'lr': 0.1 * opt.learning_rate})
    params.append({'params': [p for p in model.text_encoder.parameters()], 'lr': opt.learning_rate})
    params.append({'params': [p for p in model.transformer.parameters()], 'weight_decay': opt.weight_decay * 0.1})
    params.append({'params': [p for p in model.parameters()]})

    # remove dup params (keep the first one)
    for i1, p1 in enumerate(params):
      for i2, p2 in enumerate(params):
        if p1 is not p2:
          for p11 in p1['params']:
            for j, p22 in enumerate(p2['params']):
              if p11 is p22:
                p2['params'][j] = torch.tensor(0.0, requires_grad=True)

    optimizer = torch.optim.SGD(
        params,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay
    )
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay
        )
    return model, optimizer
                        
def compute_losses(model, data, losses_tracking, add_transformation_loss = True):
    losses = []

    # joint embedding loss
    imgs = np.stack([d['image'] for d in data])
    imgs = torch.from_numpy(imgs).float()
    if len(imgs.shape) == 2:
        imgs = model.img_encoder.fc(imgs.cuda())
    else:
         imgs = model.img_encoder(imgs.cuda())
    texts = [random.choice(d['captions']) for d in data]
    texts = model.text_encoder(texts)
    loss_name = 'joint_embedding'
    loss_weight = 1.0
    loss_value = model.pair_loss(texts, imgs).cuda()
    losses += [(loss_name, loss_weight, loss_value)]
    
    # transformation loss
    if add_transformation_loss:
        indices, source_texts, target_texts, replace_word = sample_word_pairs([d['captions'] for d in data])
        target_imgs = [imgs[i,:] for i in indices]
        target_imgs = torch.stack(target_imgs)
        source_words = [i[0] for i in replace_word]
        target_words = [i[1] for i in replace_word]

        source_texts = model.text_encoder(source_texts).detach()
        source_words = model.text_encoder(source_words).detach()
        target_texts = model.text_encoder(target_texts).detach()
        target_words = model.text_encoder(target_words).detach()
        target_imgs = target_imgs.detach()

        source_texts_to_target = model.transformer((source_texts, source_words, target_words))
        target_texts_to_source = model.transformer((target_texts, target_words, source_words))
        target_imgs_to_source = model.transformer((target_imgs, target_words, source_words))
        target_imgs_to_source_to_target = model.transformer((target_imgs_to_source, source_words, target_words))
        pairs = [
            # sources (no text dups):
            # (1) source_texts
            # (2) target_texts_to_source
            # (3) target_imgs_to_source
            #(source_texts, target_texts_to_source),
            (target_imgs_to_source, source_texts),
            #(target_imgs_to_source, target_texts_to_source),

            # targets (dups!):
            # (1) target_texts
            # (2) source_texts_to_target
            # (3) target_imgs
            # (4) target_imgs_to_source_to_target
            #(target_texts, target_imgs),
            #(target_texts, source_texts_to_target),
            (target_imgs, source_texts_to_target),
            #(target_imgs_to_source_to_target, target_imgs),

            # combination
            (torch.cat((source_texts_to_target, target_texts_to_source)), torch.cat((target_texts, source_texts))),
            (torch.cat((source_texts_to_target, target_imgs_to_source)), torch.cat((target_imgs, source_texts))),
            (torch.cat((target_imgs, target_imgs_to_source)), torch.cat((target_texts, source_texts))),
        ]
        for i, p in enumerate(pairs):
            loss_value = model.pair_loss(p[0], p[1])
            loss_name = 'loss_transformation' + str(i+1)
            loss_weight = 1.0 / len(pairs)
            losses += [(loss_name, loss_weight, loss_value)]

    # total loss
    total_loss = sum([loss_weight * loss_value for loss_name, loss_weight, loss_value in losses])
    assert(not torch.isnan(total_loss))
    losses += [('total training loss', None, total_loss)]

    # save losses
    for loss_name, loss_weight, loss_value in losses:
        if not losses_tracking.has_key(loss_name):
            losses_tracking[loss_name] = []
        losses_tracking[loss_name].append(float(loss_value.data.item()))
    return total_loss

def train_1_epoch(model, optimizer, trainset, opt, losses_tracking, add_transformation_loss = True):
    model.train()
    loader = trainset.get_loader(
        batch_size=opt.batch_size, shuffle=True,
        drop_last=True, num_workers=opt.loader_num_workers)
    for data in tqdm(loader, desc = 'training 1 epoch'):
        total_loss = compute_losses(model, data, losses_tracking, add_transformation_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

def main():
    opt = parse_opt() 
    logger = SummaryWriter(comment = opt.comment)
    print 'load datasets'
    trainset, testset, sic112 = load_datasets(opt)
    print 'create model and optimizer'
    model, optimizer = create_model(opt, trainset)
    
    # train loop
    losses_tracking = {}
    epoch = 0
    tic = time.time()
    while True:
        
        # show stat, training losses
        print 'Epoch', epoch, 'Elapsed time', round(time.time() - tic, 4)
        tic = time.time()
        for loss_name in losses_tracking:
            avg_loss = np.mean(losses_tracking[loss_name][-250:])
            print '   ', loss_name, round(avg_loss, 4)
            logger.add_scalar(loss_name, avg_loss, epoch)

        # test
        tests = []
        for dataset in [trainset, testset, sic112]:
            t = test(model, dataset, opt)
            tests += [(dataset.name() + ' ' + metric_name, metric_value) for metric_name, metric_value in t]
        for metric_name, metric_value in tests:
            print ' ', metric_name, round(metric_value, 4)
            logger.add_scalar(metric_name, metric_value, epoch)

        # train
        if epoch >= opt.num_epochs:
            break
        train_1_epoch(model, optimizer, trainset, opt, losses_tracking,
                      add_transformation_loss = epoch>=1)
        epoch += 1

        # learing rate scheduling
        if epoch % opt.learning_rate_decay_frequency == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
    # save
    torch.save({
        'model': model.cpu()
    }, opt.comment + '.pt')
 
if __name__ == '__main__':
    main() 

