
import numpy as np
import torch

def test(model, testset, opt):
    r = test_text_to_image_retrieval(model, testset, opt)
    if '112' in testset.name():
        r += test_image_to_image_word_replacement_retrieval(model, testset, opt)
    return r

def test_text_to_image_retrieval(model, testset, opt):
    model.eval()
    img_features = []
    img_labels = []
    text_features = []
    text_labels = []
    for data in testset.get_loader(batch_size = opt.batch_size, shuffle = True, drop_last= False):
        # extract image features
        imgs = np.stack([d['image'] for d in data])
        imgs = torch.from_numpy(imgs).float()
        if len(imgs.shape) == 2:
            imgs = model.img_encoder.fc(imgs.cuda())
        else:
            imgs = model.img_encoder(imgs.cuda())
        imgs = model.snorm(imgs).cpu().detach().numpy()
        img_features += [imgs]
        img_labels += [d['label'] for d in data]

        # text
        texts = []
        for d in data:
            texts += d['captions']
            text_labels += [d['label'] for c in d['captions']]
        texts = model.text_encoder(texts)
        texts = model.snorm(texts).cpu().detach().numpy()
        text_features += [texts]

        if len(img_labels) > 1100:
            break

    img_features = np.concatenate(img_features, axis=0)
    text_features = np.concatenate(text_features, axis=0)

    
    # text to image
    sims = text_features.dot(img_features.T)
    r1 = 0.0
    for i in range(sims.shape[0]):
        s = -sims[i,:]
        s = np.argsort(s)
        if text_labels[i] == img_labels[s[0]]:
            r1 += 1
    r1 /= sims.shape[0]
    return [('text2image_recall_top1', r1)]


def test_image_to_image_word_replacement_retrieval(model, testset, opt):
    model.eval()
    r = []
    
    # compute image features as retrieval database
    img_features = []
    img_labels = []
    for data in testset.get_loader(batch_size = opt.batch_size, shuffle = False, drop_last= False):
        # extract image features
        imgs = np.stack([d['image'] for d in data])
        imgs = torch.from_numpy(imgs).float()
        if len(imgs.shape) == 2:
            imgs = model.img_encoder.fc(imgs.cuda())
        else:
            imgs = model.img_encoder(imgs.cuda())
        imgs = model.snorm(imgs).cpu().detach().numpy()
        img_features += [imgs]
        img_labels += [d['label'] for d in data]
    img_features = np.concatenate(img_features, axis=0)
    
    # construct test_queries
    subjects = ['boy', 'girl', 'man', 'woman', 'dog', 'cat']
    verbs = ['sitting', 'standing', 'running', 'sleeping']
    test_sets = {
        'change_subject': [t for t in testset.test_queries_seen_objects if t['replacing_words'][0] in subjects],
        'change_verb': [t for t in testset.test_queries_seen_objects if t['replacing_words'][0] in verbs],
        'change_background': [t for t in testset.test_queries_seen_objects if t['replacing_words'][0] not in subjects+verbs],
        'change_background_novel_subject': testset.test_queries_novel_objects,
        'all': testset.test_queries_seen_objects + testset.test_queries_novel_objects
    }

    # for each test set
    for name, test_queries in test_sets.iteritems():
        query_labels = []
        for i in range(0, len(test_queries), opt.batch_size):
            query_labels += [t['target_caption'] for t in test_queries[i:(i+opt.batch_size)]]
        def measure_retrieval_performance(query_features, method_name):
            sims = query_features.dot(img_features.T)
            for k in [1, 5, 10]:
                r1 = 0.0
                for i in range(sims.shape[0]):
                    s = -sims[i,:]
                    s = np.argsort(s)
                    if query_labels[i] in [img_labels[s[j]] for j in range(k)]:
                        r1 += 1
                r1 /= sims.shape[0]
                r.append((name + '_wordreplacement_' + method_name + '_recall_top' + str(k), r1))
  
        # compute image+text query features
        query_features = []
        for i in range(0, len(test_queries), opt.batch_size):
            source_img = np.stack([img_features[t['source_idx'],:] for t in test_queries[i:(i+opt.batch_size)]])
            source_img = torch.from_numpy(source_img)
            source_words = [t['replacing_words'][0] for t in test_queries[i:(i+opt.batch_size)]]
            target_words = [t['replacing_words'][1] for t in test_queries[i:(i+opt.batch_size)]]
            source_words = model.text_encoder(source_words)
            target_words = model.text_encoder(target_words)
            f = model.snorm(model.transformer((source_img.cuda(), source_words, target_words))).cpu().detach().numpy()
            query_features += [f]
        query_features = np.concatenate(query_features, axis=0)
        measure_retrieval_performance(query_features, method_name = 'ours')

    return r








