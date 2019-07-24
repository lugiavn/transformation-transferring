import random

replace_train_list = [
    ['black', 'white', 'red', 'green', 'blue', 'pink', 'yellow', 'brown', 'gray', 'purple'],
    
    ['beach', 'park', 'mountain', 'hill', 'field',
     'street', 'road', 'city',
     'room', 'bathroom', 'kitchen', 'living room', 'house', 'building', 'restaurant'],
    
    ['people', 'boy', 'girl', 'man', 'woman', 'child', 'kid',
     'giraffe', 'dog', 'dogs', 'cat', 'cats', 'horse', 'bird', 'sheep', 'elephant',
     'truck', 'car', 'bus', 'train', 'bike', 'motorcycle', 'plane', 'airplane'],
    
    ['table', 'plate', 'bowl', 'chair', 'bench', 'couch', 'bed',
     'sink', 'toilet',
     'food', 'pizza', 'sandwich', 'cake', 
     'computer', 'laptop', 'board', 'phone',
     'skateboard', 'surfboard', 'frisbee', 'kite', 'baseball', 'tennis', 'frisbee', 'bat',
     'umbrella','shirt', 'jacket', 'hat'],
    
    ['play', 'playing', 'run', 'running', 'sleep', 'sleeping', 'walk', 'walking', 
     'talk', 'talking', 'ride', 'riding', 'eat', 'eating', 'fly', 'flying',
     'sits', 'sit', 'sitting', 'seating', 'standing', 'holding', 'wearing', 'laying'],
    
    ['sandy', 'grassy', 'wet', 'wooden'],
    ['small', 'big', 'little', 'young', 'old'],
    ['sand', 'grass', 'water', 'snow', 'tree', 'hydrant', 'sky', 'fire', 'ocean'],
    
]

all_words = []
for l in replace_train_list:
    all_words += l

def find_a_text_to_replace(texts):
    w_found = False
    random.shuffle(texts)
    for text in texts:
        random.shuffle(replace_train_list)
        for l in replace_train_list:
            random.shuffle(l)
            for w in l:
                if w in text.split():
                    w2 = w
                    while w2 == w:
                        if random.random() > 0.1:
                            w2 = random.choice(l)
                        else:
                            w2 = random.choice(all_words)
                    w_found = True
                if w_found:
                    break
            if w_found:
                break
        if w_found:
            break

    if w_found:
        t = ' '.join([w2 if w == j else j for j in text.split()])
        source_text = t
        target_text = random.choice(texts)
        replace_word = (w2, w)
        return source_text, target_text, replace_word
    return None, None, None