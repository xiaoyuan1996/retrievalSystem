"""
对用户提供的图像编码和文本编码服务进行封装和测试
"""

import numpy as np
import sys
sys.path.append("..")
from models import encoder
import logging
import globalvar

# 变量初始化
logger = globalvar.get_value("logger")



def image_encoder_api(model, image_path):
    """
    对用户提供的图像编码函数进行二次封装
    :param model: 用户提供的模型
    :param image_path: 提供的图片路径
    :return: 编码向量 512数组
    """
    # logger.info("Encode image: {}".format(image_path))
    image_vector = encoder.image_encoder(model, image_path)
    return image_vector

def text_encoder_api(model, vocab, text):
    """
    对用户提供的文本编码函数进行二次封装
    :param model: 用户提供的模型
    :param vocab: 提供的字典
    :param text: 提供的文本
    :return: 编码向量 512数组
    """
    # logger.info("Encode text: {}".format(text))
    text_vector = encoder.text_encoder(model, vocab, text)
    return text_vector

def l2norm(X):
    """L2-normalize columns of X
    """
    return X/np.linalg.norm(X)

def cosine_sim_api(image_vector, text_vector):
    """
    计算两个向量间的余弦相似度
    :param image_vector: 图片编码向量
    :param text_vector: 文本编码向量
    :return: 相似度
    """
    image_vector = l2norm(image_vector)
    text_vector = l2norm(text_vector)

    similarity = np.mean(np.multiply(image_vector, text_vector))
    return similarity

def test_function_api(model, vocab, image_path, text):
    """
    测试基本服务运行正常
    :param model: 用户提供的模型
    :param vocab: 提供的字典
    :param image_path: 提供的图片路径
    :param text: 提供的文本
    :return: None
    """
    logger.info("Test base function is running successfully ...")
    try:
        image_vector = image_encoder_api(model, image_path)
        text_vector = text_encoder_api(model, vocab, text)
        similarity = cosine_sim_api(image_vector, text_vector)
        logger.info("Base function running successfully.\n")
    except Exception as e:
        logger.error("Base function running failed: {}\n".format(e))
        exit(0)

