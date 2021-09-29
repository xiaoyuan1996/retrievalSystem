from api_controlers import base_function
from api_controlers import utils
import sys,time
sys.path.append("..")
import globalvar

# 变量初始化
model = globalvar.get_value("model")
vocab_word = globalvar.get_value("vocab_word")
logger = globalvar.get_value("logger")


def image_search(request_data):
    logger.info("Request json: {}".format(request_data))

    # 检测请求完备性
    if not isinstance(request_data, dict):
        return utils.get_stand_return(False, "Request must be dicts, and have keys: image_path, retrieved_ids, start, end.")
    if 'image_path' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: str image_path.")
    if 'retrieved_ids' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: list retrieved_ids, default = *.")
    if 'start' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: int start.")
    if 'end' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: int end.")

    # 解析
    image_path, retrieved_ids, start, end = request_data['image_path'],  request_data['retrieved_ids'], request_data['start'], request_data['end']

    #编码文本
    image_vector = base_function.image_encoder_api(model, image_path)

    # 向量比对
    logger.info("Parse request correct, start image retrieval ...")
    time_start = time.time()

    retrieval_results = {}
    rsd = globalvar.get_value("rsd")
    if retrieved_ids == "*":  # 检索所有影像
        for k in rsd.keys():
            retrieval_results[k] = base_function.cosine_sim_api(image_vector, rsd[k])
    else:
        for k in retrieved_ids: # 检索指定影像
            retrieval_results[k] = base_function.cosine_sim_api(image_vector, rsd[k])
    sorted_keys = utils.sort_based_values(retrieval_results)[start:end] # 排序

    time_end = time.time()
    logger.info("Retrieval finished in {:.4f}s.".format(time_end - time_start))
    return utils.get_stand_return(True, sorted_keys)


