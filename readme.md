# Backend of cross-modal retrieval system
##### Author: Zhiqiang Yuan 


### -------------------------------------------------------------------------------------
### Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then we'll let you know when we update

The back-end of cross-modal retrieval system，wihch will contain services such as semantic location .etc .
The purpose of this project is to provide a set of applicable retrieval framework for the retrieval model.
We will use RS image data as the baseline for development, and demonstrate the potential of the project through services such as semantic positioning and cross-modal retrieval.

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
### Environments
## Under Updating
