```
git clone https://github.com/ultralytics/yolov5  # clone 
cd yolov5
pip install -qr requirements.txt comet_ml  # install
pip install ultralytics
```
```
import torch
import utils
display = utils.notebook_init()  # checks
```
```
!python train.py img=640 batch=64 epochs=300 data=/content/dataRgb/data.yaml cfg=/content/yolov5/models/magno.yaml weights='' hyp=/content/custom_hyp.yaml project=runs/train name=magnoatch
```
