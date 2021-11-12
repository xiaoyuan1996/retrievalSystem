import requests, json

def post_encode_image():
    # 编码请求
    # image_id: file_path
    data = [
        {
            "image_id": 12,
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
    url = 'http://192.168.140.241:33133/api/image_encode/'

    r = requests.post(url, data=json.dumps(data))
    print(r.json())

def post_delete_encode():
    # 删除编码请求
    # image_id
    data = {"deleteID":"32"}
    url = 'http://192.168.140.241:33133/api/delete_encode/'

    r = requests.post(url, data=json.dumps(data))
    print(r.json())

def post_t2i_rerieval():
    # 文本检索请求
    # text
    data = {
        'text': "One block has a cross shaped roof church.",
        'user_id': 2,
        'page_no': 2,
        'page_size': 0
    }
    url = 'http://192.168.140.241:33133/api/text_search/'

    r = requests.post(url, data=json.dumps(data))
    print(r.json())

def post_i2i_retrieval():
    # 图像检索请求
    # image
    data = {
        'image_path': "../data/test_data/images/00013.jpg",
        'user_id': 1,
        'page_no': 1,
        'page_size': 10
    }
    url = 'http://192.168.140.241:33133/api/image_search/'

    r = requests.post(url, data=json.dumps(data))
    print(r.json())

def post_semantic_localization():
    # 语义定位请求
    # # semantic localization
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
    url = 'http://192.168.140.241:33133/api/semantic_localization/'

    r = requests.post(url, data=json.dumps(data))
    print(r.json())

if __name__=="__main__":
    # post_semantic_localization()
    # post_encode_image()
   # post_delete_encode()
   # post_t2i_rerieval()
    post_i2i_retrieval()
