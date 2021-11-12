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
        return utils.get_stand_return(False, "Request must be dicts, and have keys: image_path, user_id, page_no, page_size.")
    if 'image_path' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: str image_path.")
    if 'user_id' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: int user_id")
    if 'page_no' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: int page_no.")
    if 'page_size' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: int page_size.")

    # 解析
    image_path, user_id, page_no, page_size = request_data['image_path'],  int(request_data['user_id']), int(request_data['page_no']), int(request_data['page_size'])

    # 出错机制
    if page_no <=0 or page_size <=0 :
        return utils.get_stand_return(False, "Request page_no and page_size must >= 1.")

    #编码图像
    image_vector = base_function.image_encoder_api(model, image_path)

    # 向量比对
    logger.info("Parse request correct, start image retrieval ...")
    time_start = time.time()

    # 统计匹配数据
    rsd = globalvar.get_value("rsd")
    rsd_retrieved, retrieval_results = {}, {}
    for k,v in rsd.items():
        if (rsd[k]["privilege"] == 1) or (rsd[k]["user_id"] == user_id):
            rsd_retrieved[k] = v

    # 计算
    for k in rsd_retrieved.keys():
        retrieval_results[k] = base_function.cosine_sim_api(image_vector, rsd[k]["image_vector"])

    # 排序
    start, end = page_size * (page_no-1), page_size * page_no
    sorted_keys = utils.sort_based_values(retrieval_results)[start:end]

    time_end = time.time()
    logger.info("Retrieval finished in {:.4f}s.".format(time_end - time_start))
    return utils.get_stand_return(True, sorted_keys)


