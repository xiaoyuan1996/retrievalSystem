from api_controlers import base_function
from api_controlers import utils
import sys,time
sys.path.append("..")
import globalvar

# 变量初始化
model = globalvar.get_value("model")
vocab_word = globalvar.get_value("vocab_word")
logger = globalvar.get_value("logger")


def semantic_localization_runner(request_data):
    logger.info("\nRequest json: {}".format(request_data))

    # 图像切割，保存
    # 分别编码
    # 文本编码
    # 合并生热力图
    # 返回路径或图像

    # time.sleep(10)
    image_vector = base_function.image_encoder_api(model, "../data/test_data/images/00013.jpg")
    print(image_vector)
    print(type(image_vector))
    # text_vector = base_function.text_encoder_api(model, vocab_word, "One block has a cross shaped roof church.")
    # sims = base_function.cosine_sim_api(image_vector, text_vector)
    # print(sims)

    logger.info("Encode successful for request: {}\n".format(request_data))
    return utils.get_stand_return(True, "encode successful")