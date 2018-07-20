### Multispectral Deep Neural Networks for Pedestrian Detection

This repository is a folk of multisepectral deepnet for pedetstrian detection [code](https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn). For how to install the required softwares and set up the code in right configuration, e.g., Caffe, pycaffe, please refer to the most original [README.md](https://github.com/rbgirshick/py-faster-rcnn/blob/master/README.md).

### Download pretrained models
[VGG16 model on caltech](https://drive.google.com/open?id=0ByrJI3mShdW6WVBxQldmdnE2S2s) trained on Caltech pedestrian dataset.

[VGG16 model on kaist (RGB input)](https://drive.google.com/open?id=0ByrJI3mShdW6LWNqT0tYQ3JteW8) trained on Kaist pedestrian dataset.

[VGG16 model on kaist (multispectral input)](https://drive.google.com/open?id=0ByrJI3mShdW6R3R1dkE4QlNQUUk) trained on Kaist multispectral dataset.

Save these models to `models/caltech/VGG16/`, `models/kaist/VGG16/`, and `models/kaist_fusion/VGG16/`, respectively.

### Run demos
Run `sh ./run_demo.sh caltech` for images from Caltech.

Run `sh ./run_demo.sh kaist-color` for images from Kaist.

Run `sh ./run_demo.sh kaist-fusion` for multispectral images from Kaist.

### train


### test
Save the test data in `data/kaist/test_all/`.
In the root directory, Run `./tools/test_net.py 
--gpu 0  --def models/kaist_fusion/VGG16/faster_rcnn_test.pt  
--net models/kaist_fusion/VGG16/VGG16_faster_rcnn_final_kaist_fusion.caffemodel --cfg experiments/cfgs/faster_rcnn_end2end_fusion.yml --imdb kaist_test-all`
for test the pretrianed kaist_fusion model.
