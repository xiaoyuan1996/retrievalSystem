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


# 加入影像编码池
def image_encode_append(request_data):
    logger.info("Request json: {}".format(request_data))

    # 加入未编码数据
    for item in request_data:
        unembeded_images = globalvar.get_value("unembeded_images")
        unembeded_images = utils.dict_insert(int(item["image_id"]), item, unembeded_images)
        globalvar.set_value("unembeded_images", value=unembeded_images)

    logger.info("Request append successfully for above request.\n")
    return utils.get_stand_return(True, "Image append successful")

# 检查是否有影像未编码
def image_encode_runner():

    unembeded_images = globalvar.get_value("unembeded_images")

    if unembeded_images != {}:
        logger.info("{} images in unembeded image pool have been detected ...".format(len(unembeded_images.keys())))
        for img_id in list(unembeded_images.keys()):
            img_path = unembeded_images[img_id]["image_path"]

            image_vector = base_function.image_encoder_api(model, img_path)

            # 更新rsd数据
            rsd = globalvar.get_value("rsd")
            unembeded_images[img_id]["image_vector"] = image_vector
            rsd = utils.dict_insert(img_id, unembeded_images[img_id], rsd)

            globalvar.set_value("rsd", value=rsd)

            # 删除未编码池
            unembeded_images = utils.dict_delete(img_id, unembeded_images)
            globalvar.set_value("unembeded_images", value=unembeded_images)

        # 保存rsd数据
        utils.dict_save(rsd, cfg['data_paths']['rsd_path'])

        logger.info("Embeding images pool successfully.")

    # 进行下一次定时任务
    check_unembeded_image = Timer(5, image_encode_runner)
    check_unembeded_image.start()
