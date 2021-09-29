import sys
sys.path.append("..")
from models import init as model_init
import globalvar

# 变量初始化
logger = globalvar.get_value("logger")

def init_model(cfg_models):
    """
    对用户提供的模型代码进行二次封装
    :param cfg_models: 关于模型的配置信息
    :return: model: 创建好的模型
    """
    logger.info("Model init ...")
    try:
        model, vocab_word = model_init.model_init(cfg_models['prefix_path'])
        model.eval()
        logger.info("Model init successfully.\n")
        return model, vocab_word
    except Exception as e:
        logger.error("Model init failed: {}\n".format(e))
        exit(0)