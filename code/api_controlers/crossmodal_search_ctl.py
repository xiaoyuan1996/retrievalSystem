from api_controlers import base_function
from api_controlers import utils
import sys,time
sys.path.append("..")
import globalvar

# 变量初始化
model = globalvar.get_value("model")
vocab_word = globalvar.get_value("vocab_word")
logger = globalvar.get_value("logger")


def crossmodal_search(request_data):
    logger.info("Request json: {}".format(request_data))

    # 检测请求完备性
    if not isinstance(request_data, dict):
        return utils.get_stand_return(False, "Request must be dicts, and have keys: text, retrieved_ids, start, end.")
    if 'text' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: str text.")
    if 'retrieved_ids' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: list retrieved_ids, default = *.")
    if 'start' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: int start.")
    if 'end' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: int end.")

    # 解析
    text, retrieved_ids, start, end = request_data['text'],  request_data['retrieved_ids'], request_data['start'], request_data['end']

    #编码文本
    text_vector = base_function.text_encoder_api(model, vocab_word, text)

    # 向量比对
    logger.info("Parse request correct, start retrieval ...")
    time_start = time.time()

    retrieval_results = {}
    rsd = globalvar.get_value("rsd")
    if retrieved_ids == "*":  # 检索所有影像
        for k in rsd.keys():
            retrieval_results[k] = base_function.cosine_sim_api(text_vector, rsd[k])
    else:
        for k in retrieved_ids: # 检索指定影像
            retrieval_results[k] = base_function.cosine_sim_api(text_vector, rsd[k])
    sorted_keys = utils.sort_based_values(retrieval_results)[start:end] # 排序

    time_end = time.time()
    logger.info("Retrieval finished in {:.4f}s.".format(time_end - time_start))
    return utils.get_stand_return(True, sorted_keys)


