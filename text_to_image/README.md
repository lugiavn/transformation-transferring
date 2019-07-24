# Let's transfer text transformations to images
![ZZ](../cocoqual1.png?raw=true "X")

## Datasets

Download the MSCOCO 2014 dataset from http://cocodataset.org
```
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```
Unzip into ./datasets/coco. Make sure the files are:
```
./datasets/coco/annotations/...json
./datasets/coco/train2014/...jpg
./datasets/coco/val2014/...jpg
```
Download SIC112: refer to SIC112.md
```
./datasets/SIC112/caption/image
```


## Run stuff

Clone google/tirg code here:
```
git clone http://github.com/google/tirg

```

Then run the train+test:
```
python main.py --coco_path ./datasets/coco --sic112_path ./datasets/SIC112

```
The first run will take a couple of hours to cache the image features. After that, training starts and would go for another hour to get good result.

Logs will be saved at ./runs, monitor with tensorboard

The final model will be saved as "x.pt".

Use play_with_saved_model.ipynb notebook to inspect some qualitative result.
