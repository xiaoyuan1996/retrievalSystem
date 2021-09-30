import requests, json

def post_encode_image():
    # 编码请求
    # image_id: file_path
    data = {
        11:"../data/test_data/images/00013.jpg",
        33: "../data/test_data/images/00013.jpg",
        32: "../data/test_data/images/00013.jpg",
    }
    url = 'http://192.168.43.216:49205/api/image_encode/'

    r = requests.post(url, data=json.dumps(data))
    print(r.json())

def post_delete_encode():
    # 删除编码请求
    # image_id
    data = ['3']
    url = 'http://192.168.43.216:49205/api/delete_encode/'

    r = requests.post(url, data=json.dumps(data))
    print(r.json())

def post_t2i_rerieval():
    # 文本检索请求
    # text
    data = {
        'text': "One block has a cross shaped roof church.",
        'retrieved_ids': "*",
        'start': 0,
        'end': 100
    }
    url = 'http://192.168.43.216:49205/api/text_search/'

    r = requests.post(url, data=json.dumps(data))
    print(r.json())

def post_i2i_retrieval():
    # 图像检索请求
    # image
    data = {
        'image_path': "../data/test_data/images/00013.jpg",
        'retrieved_ids': "*",
        'start': 0,
        'end': 100
    }
    url = 'http://192.168.43.216:49205/api/image_search/'

    r = requests.post(url, data=json.dumps(data))
    print(r.json())

def post_semantic_localization():
    # 语义定位请求
    # # semantic localization
    data = {
        'image_path': "../data/test_data/images/demo1.tif",
        'text': "there are two tennis courts beside the playground",
        'params': {
            'steps': [128,256,512]
        },
    }
    url = 'http://192.168.43.216:49205/api/semantic_localization/'

    r = requests.post(url, data=json.dumps(data))
    print(r.json())

if __name__=="__main__":
    post_semantic_localization()