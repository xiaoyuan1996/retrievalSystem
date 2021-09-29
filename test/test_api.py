import requests, json

# # 编码请求
# # image_id: file_path
# data = {
#     11:"../data/test_data/images/00013.jpg",
#     33: "../data/test_data/images/00013.jpg",
#     32: "../data/test_data/images/00013.jpg",
# }
# url = 'http://192.168.43.216:49205/api/image_encode/'
#
# r = requests.post(url, data=json.dumps(data))
# print(r.json())


# # 删除编码请求
# # image_id
# data = ['3']
# url = 'http://192.168.43.216:49205/api/delete_encode/'
#
# r = requests.post(url, data=json.dumps(data))
# print(r.json())


# # 文本检索请求
# # text
# data = {
#     'text': "One block has a cross shaped roof church.",
#     'retrieved_ids': "*",
#     'start': 0,
#     'end': 100
# }
# url = 'http://192.168.43.216:49205/api/text_search/'
#
# r = requests.post(url, data=json.dumps(data))
# print(r.json())

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