import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import nltk
import numpy as np
import yaml
import argparse
from PIL import Image
import cv2

def get_image_size(img_path):
    """
    获取图像大小
    :param img_path: 图片地址
    :return: Mb
    """
    return os.path.getsize(img_path)

def trans_bigimage_to_small(bigimg_path, threshold=50):
    """
    转换大型图像到小型图像
    :param bigimg_path: 大型图片地址
    :param threshold: 阈值 50 * 1024 即 50Mb
    :return: 小型图像地址
    """
    if get_image_size(bigimg_path) <= threshold:
        return bigimg_path
    else:
        image = cv2.imread(bigimg_path)
        image = cv2.resize(image, (256, 256))
        cv2.imwrite("tmp/tmp.jpg", image)
        return "tmp/tmp.jpg"


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def image_encoder(model, image_path):
    """
    提供的图像编码函数
    :param model: 模型文件
    :param image_path: 图像路径
    :return: 编码向量
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    # check image size
    image_path = trans_bigimage_to_small(image_path)

    # data preprocessing
    image = Image.open(image_path).convert('RGB')
    image = transform(image)  # torch.Size([3, 256, 256])
    image = torch.unsqueeze(image, dim=0).cuda()

    # model processing
    lower_feature, higher_feature, solo_feature = model.extract_feature(image)
    global_feature = model.mvsa(lower_feature, higher_feature, solo_feature)
    global_feature = l2norm(global_feature, dim=-1)

    # to cpu vector
    vector = global_feature.cpu().detach().numpy()[0]

    return vector

def text_encoder(model, vocab, text):
    """
    提供的文本编码函数
    :param model: 模型文件
    :param vocab: 文本字典
    :param text: 编码文本
    :return: 编码向量
    """

    # Convert caption (string) to word ids.
    tokens = nltk.tokenize.word_tokenize(text.lower())
    punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    tokens = [k for k in tokens if k not in punctuations]
    tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]

    caption = []
    caption.extend([vocab(token) for token in tokens_UNK])
    caption = torch.LongTensor(caption)
    caption = torch.unsqueeze(caption, dim=0).cuda()
    caption = caption.expand((2,caption.shape[-1]))

    # model processing
    text_feature = model.text_feature(caption)
    text_feature = l2norm(text_feature, dim=-1)

    # to cpu vector
    vector = text_feature.cpu().detach().numpy()[0]

    return vector

