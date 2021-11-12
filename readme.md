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
flask >= 1.1.1
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
   
data = [
    {
        "image_id": 11,
        "image_path": "../data/test_data/images/00013.jpg",
        "user_id": 1,
        "privilege": 1
    },
    {
        "image_id": 33,
        "image_path": "../data/test_data/images/00013.jpg",
        "user_id": 1,
        "privilege": 1
    },
    {
        "image_id": 32,
        "image_path": "../data/test_data/images/00013.jpg",
        "user_id": 2,
        "privilege": 1
    }
]
url = 'http://192.168.97.241:33133/api/image_encode/'

r = requests.post(url, data=json.dumps(data))
print(r.json())
```

```bash
------------------------------------------
#/api/delete_encode/ [POST]  
# FUNC: delete encodes
   
# image_id
data = {"deleteID":"32"}
url = 'http://192.168.140.241:33133/api/delete_encode/'

r = requests.post(url, data=json.dumps(data))
print(r.json())
```

```bash
------------------------------------------
#/api/text_search/ [POST]  
# FUNC: cross-modal retrieval 
   
data = {
    'text': "One block has a cross shaped roof church.",
    'user_id': 1,
    'page_no': 1,
    'page_size': 10
}
url = 'http://0.0.0.0:33133/api/text_search/'

r = requests.post(url, data=json.dumps(data))
print(r.json())
```

```bash
------------------------------------------
#/api/image_search/ [POST]  
# FUNC: image-image retrieval 
   
data = {
    'image_path': "/data/test_data/images/00013.jpg",
    'user_id': 1,
    'page_no': 1,
    'page_size': 10
}
url = 'http://0.0.0.0:33133/api/image_search/'

r = requests.post(url, data=json.dumps(data))
print(r.json())
```

```bash
------------------------------------------
#/api/semantic_localization/ [POST]  
# FUNC: semantic localization
   
data = {
     "input_file": ["../data/test_data/images/demo1.tif"],
     "output_file": [
        "../data/retrieval_system_data/semantic_localization_data/heatmap.png",
        "../data/retrieval_system_data/semantic_localization_data/heatmap_add.png"],
     "params":  {
                  "text": "there are two tennis courts beside the playground",
                  "steps": [128,256,512]
                }
       }
url = 'http://192.168.97.241:33133/api/semantic_localization/'

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

```
RUN:
2021-11-12 14:43:39,447 - __main__ - INFO - Loading config file from common/config.yaml
2021-11-12 14:43:39,450 - __main__ - INFO - Create init variables
2021-11-12 14:43:41,242 - __main__ - INFO - Model init ...
Warning: 61/930911 words are not in dictionary, thus set UNK
2021-11-12 14:43:49,147 - __main__ - INFO - Model init successfully.
2021-11-12 14:43:49,149 - __main__ - INFO - Test base function is running successfully ...
2021-11-12 14:43:50,693 - __main__ - INFO - Base function running successfully.
2021-11-12 14:43:51,492 - __main__ - INFO - Start apis and running ...
2021-11-12 14:43:51,503 - werkzeug - INFO -  * Running on http://192.168.140.241:33133/ (Press CTRL+C to quit)
 * Serving Flask app "api_controlers.apis" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
```

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


