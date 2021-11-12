from flask import Flask, request, jsonify
import json, os
from threading import Timer

from api_controlers import base_function, image_encode_ctl, delete_encode_ctl,\
    text_search_ctl, image_search_ctl, semantic_localization_ctl,  utils

def api_run(cfg):
    app = Flask(__name__)  # Flask 初始化
    # ======================= 接口定义 ============================

    # 影像编码
    @app.route(cfg['apis']['image_encode']['route'], methods=['post'])
    def image_encode():
        request_data = json.loads(request.data.decode('utf-8'))
        return_json = image_encode_ctl.image_encode_append(request_data)
        return return_json

    # 删除编码
    @app.route(cfg['apis']['delete_encode']['route'], methods=['post'])
    def delete_encode():
        request_data = json.loads(request.data.decode('utf-8'))
        return_json = delete_encode_ctl.delete_encode(request_data)
        return return_json

    # 文本检索
    @app.route(cfg['apis']['text_search']['route'], methods=['post'])
    def text_search():
        request_data = json.loads(request.data.decode('utf-8'))
        return_json = text_search_ctl.text_search(request_data)
        return return_json

    # 图像检索
    @app.route(cfg['apis']['image_search']['route'], methods=['post'])
    def image_search():
        request_data = json.loads(request.data.decode('utf-8'))
        return_json = image_search_ctl.image_search(request_data)
        return return_json

    # 语义定位
    @app.route(cfg['apis']['semantic_localization']['route'], methods=['post'])
    def semantic_localization():
        request_data = json.loads(request.data.decode('utf-8'))
        return_json = semantic_localization_ctl.semantic_localization(request_data)
        return return_json

    # 定时任务
    # 循环检测待编码池中是否还有未编码数据
    check_unembeded_image = Timer(5, image_encode_ctl.image_encode_runner)
    check_unembeded_image.start()

    app.run(host=cfg['apis']['hosts']['ip'], port=cfg['apis']['hosts']['port'])
