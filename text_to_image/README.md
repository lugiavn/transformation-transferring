# Let's transfer text transformations to images
![ZZ](../cocoqual1.png?raw=true "X")

...

First clone google/tirg here:

git clone http://github.com/google/tirg

Then run the train+test:

python main.py --coco_path ../../../datasets/coco --sic112_path ../../SIC112

logs will be saved at ./runs, monitor with tensorboard
