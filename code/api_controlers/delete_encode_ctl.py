from api_controlers import base_function
from api_controlers import utils
import sys,time
from threading import Timer

sys.path.append("..")
import globalvar

# 变量初始化
model = globalvar.get_value("model")
vocab_word = globalvar.get_value("vocab_word")
logger = globalvar.get_value("logger")
cfg = utils.get_config()

def delete_encode(request_data):
    logger.info("\nRequest json: {}".format(request_data))


    request_data = [int(i) for i in request_data['deleteID'].split(",")]
    if request_data != []:
        # 删除数据
        for k in request_data:
            rsd = globalvar.get_value("rsd")
            if k not in rsd.keys():
                return utils.get_stand_return(False, "Key {} not found in encode pool.".format(k))
            else:
                rsd = utils.dict_delete(int(k), rsd)
            globalvar.set_value("rsd", rsd)

        utils.dict_save(rsd, cfg['data_paths']['rsd_path'])

        logger.info("Request delete successfully for above request.\n")
    return utils.get_stand_return(True, "Image delete successful")
