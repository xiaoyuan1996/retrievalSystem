import globalvar
from api_controlers import utils
import logging
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 记录设置
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    globalvar.set_value("logger", logger)

    # 加载配置文件
    cfg_path = "common/config.yaml"
    logger.info("Loading config file from {}".format(cfg_path))
    cfg = utils.get_config(cfg_path)

    # 创建起始变量
    logger.info("Create init variables")
    globalvar.set_value("unembeded_images", value={})
    globalvar.set_value("rsd", value=utils.init_rsd(cfg['data_paths']['rsd_dir_path']))
    utils.create_dirs(cfg['data_paths']['rsd_path'])
    utils.create_dirs(cfg['data_paths']['semantic_localization_path'])
    utils.create_dirs(cfg['data_paths']['temp_path'])

    # 模型初始化
    from api_controlers import model_init_ctl
    model, vocab_word = model_init_ctl.init_model(cfg['models'])
    globalvar.set_value('model', model)
    globalvar.set_value('vocab_word', vocab_word)

    # 测试模型是否编码正常
    from api_controlers import base_function
    base_function.test_function_api(model, vocab_word, cfg['launch_test']['test_image'], cfg['launch_test']['test_text'])

    # 开启接口
    from api_controlers import apis
    logger.info("Start apis and running ...\n")
    apis.api_run(cfg)