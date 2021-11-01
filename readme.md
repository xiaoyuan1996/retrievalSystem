# Backend of Cross-modal Retrieval System
##### Author: Zhiqiang Yuan 

<a href="https://github.com/xiaoyuan1996/retrievalSystem"><img src="https://travis-ci.org/Cadene/block.bootstrap.pytorch.svg?branch=master"/></a>
![Supported Python versions](https://img.shields.io/badge/python-3.7-blue.svg)
![Supported OS](https://img.shields.io/badge/Supported%20OS-Linux-yellow.svg)
![npm License](https://img.shields.io/npm/l/mithril.svg)
<a href="https://pypi.org/project/mitype/"><img src="https://img.shields.io/pypi/v/mitype.svg"></a>

### -------------------------------------------------------------------------------------
### Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then we'll let you know when we update

The back-end of cross-modal retrieval system，wihch will contain services such as semantic location .etc .
The purpose of this project is to provide a set of applicable retrieval framework for the retrieval model.
We will use RS image data as the baseline for development, and demonstrate the potential of the project through services such as semantic positioning and cross-modal retrieval.


#### Summary

* [Requirements](#requirements)
* [Apis](#apis)
* [Architecture](#architecture)
* [Three Steps to Use This Framework](#three-steps-to-use-this-framework)
* [Customize Your Rerieval Model](#customize-your-rerieval-model)
* [Citation](#citation)
### -------------------------------------------------------------------------------------
### Requirements
```bash
numpy>=1.7.1
six>=1.1.0
PyTorch > 0.3
flash >= 1.1.1
Numpy
h5py
nltk
yaml
```
------------------------------------------

### -------------------------------------------------------------------------------------
### Apis
```bash
------------------------------------------
#/api/image_encode/ [POST]  
# FUNC: encode images
   
data = {
 # image_id: file_path
 11:"../data/test_data/images/00013.jpg",
 33: "../data/test_data/images/00013.jpg",
 32: "../data/test_data/images/00013.jpg",
}
url = 'http://192.168.43.216:49205/api/image_encode/'

r = requests.post(url, data=json.dumps(data))
print(r.json())
```

```bash
------------------------------------------
#/api/delete_encode/ [POST]  
# FUNC: delete encodes
   
# image_id
data = [3, 4]
url = 'http://192.168.43.216:49205/api/delete_encode/'
r = requests.post(url, data=json.dumps(data))
print(r.json())
```

```bash
------------------------------------------
#/api/text_search/ [POST]  
# FUNC: cross-modal retrieval 
   
data = {
     'text': "One block has a cross shaped roof church.",  # retrieved text
     'retrieved_ids': "*",  # retrieved images pool
     'start': 0,    # from top
     'end': 100     # to end
 }
url = 'http://192.168.43.216:49205/api/text_search/'
r = requests.post(url, data=json.dumps(data))
print(r.json())
```

```bash
------------------------------------------
#/api/image_search/ [POST]  
# FUNC: image-image retrieval 
   
data = {
     'image_path': "../data/test_data/images/00013.jpg",,  # retrieved image
     'retrieved_ids': "*",  # retrieved images pool: 1) * represents all, 2) [1, 2, 4] represent images pool
     'start': 0,    # from top
     'end': 100     # to end
 }
url = 'http://192.168.43.216:49205/api/image_search/'
r = requests.post(url, data=json.dumps(data))
print(r.json())
```

```bash
------------------------------------------
#/api/semantic_localization/ [POST]  
# FUNC: semantic localization
   
data = {
    'image_path': "../data/test_data/images/demo1.tif",
    'text': "there are two tennis courts beside the playground",
    'params': {
        'steps': [64, 128,256,512]
    },
}
url = 'http://192.168.43.216:49205/api/semantic_localization/'
r = requests.post(url, data=json.dumps(data))
print(r.json())
```

### -------------------------------------------------------------------------------------
### Architecture

```bash
-- code     # all codes
    -- api_controls     # control files
    -- common           # config file
    -- models           # put the retrieval mdoel here
    -- globalvar.py     # global varibles define
    -- main.py          # main file

-- data
    -- retrieval_system_data    # project data here
    -- test_data        # image database here

-- figure   # some figures about this project

-- test     # test function
```

### -------------------------------------------------------------------------------------
### Three Steps to Use This Framework

Step 1. Install the environment, download the code to the local, and change the path setting of the ./code/common/config file. At the same time, you need to change the yaml path file under ./code/models/options/ .

Step 2. Enter the ./code directory and run main.py to start the flask service.

Step 3. Use Postman etc. or python's built-in request service for sample requests. Some interface samples have been shown in ./test/test_qpi.py .


### -------------------------------------------------------------------------------------
### Customize Your Rerieval Model

You only need to change the ./code/models folder to make your retrieval model run in the service. For this, you should provide encoding interfaces and model initialization interfaces for different modal data. For more information about this, please see the README file under ./code/models/ .

## Under Updating

## Citation
If you feel this code helpful or use this code or dataset, please cite it as
```
Z. Yuan et al., "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3078451.

Z. Yuan et al., "A Lightweight Multi-scale Crossmodal Text-Image Retrieval Method In Remote Sensing," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3124252.
```


