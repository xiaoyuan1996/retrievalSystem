from api_controlers import base_function
from api_controlers import utils
import sys,time
import cv2
import numpy as np
import os
sys.path.append("..")
import globalvar

# 变量初始化
model = globalvar.get_value("model")
vocab_word = globalvar.get_value("vocab_word")
logger = globalvar.get_value("logger")
cfg = utils.get_config()


def split_image(img_path, steps):
    subimage_files_dir = os.path.join(cfg['data_paths']['temp_path'], os.path.basename(img_path).split(".")[0])

    # 裁切图像文件夹
    subimages_dir = subimage_files_dir +'_subimages'
    if os.path.exists(subimages_dir):
        utils.delete_dire(subimages_dir)
    else:
        os.makedirs(subimages_dir)

    # Read Image
    source_img = cv2.imread(img_path)
    img_weight = np.shape(source_img)[0]
    img_height = np.shape(source_img)[1]
    logger.info("img size:{}x{}".format(img_weight, img_height))

    for step in steps:
        logger.info("Start split images with step {}".format(step))
        for gap in [step, 0.5 * step]:
            gap = int(gap)

            # Cut img
            for h in range(0 + (step - gap), img_height, step):
                h_start, h_end = h, h + step
                # bound?
                if h_end >= img_height:
                    h_start, h_end = img_height - step, img_height

                for w in range(0 + (step - gap), img_weight, step):
                    w_start, w_end = w, w + step
                    # bound?
                    if w_end >= img_weight:
                        w_start, w_end = img_weight - step, img_weight

                    cut_img_name = str(w_start) + "_" + str(w_end) + "_" + str(h_start) + "_" + str(h_end) + ".jpg"
                    cut_img = source_img[w_start:w_end, h_start:h_end]
                    cut_img = cv2.resize(cut_img, (256, 256), interpolation=cv2.INTER_CUBIC)

                    cv2.imwrite(os.path.join(subimages_dir, cut_img_name), cut_img)


    logger.info("Image {} has been split successfully.".format(img_path))

def generate_heatmap(img_path, text, output_file_h, output_file_a):
    subimages_dir = os.path.join(cfg['data_paths']['temp_path'], os.path.basename(img_path).split(".")[0]) +'_subimages'

#    heatmap_subdir = utils.create_random_dirs_name(cfg['data_paths']['temp_path'])
#    heatmap_dir = os.path.join(cfg['data_paths']['semantic_localization_path'], heatmap_subdir)
#    heatmap_dir = output_path

    # 清除缓存
#    if os.path.exists(heatmap_dir):
#        utils.delete_dire(heatmap_dir)
#    else:
#        os.makedirs(heatmap_dir)

    logger.info("Start calculate similarities ...")
    cal_start = time.time()

    # text vector
    text_vector = base_function.text_encoder_api(model, vocab_word, text)

    # read subimages
    subimages = os.listdir(subimages_dir)
    sim_results = []
    for subimage in subimages:
        image_vector = base_function.image_encoder_api(model, os.path.join(subimages_dir, subimage))
        sim_results.append(base_function.cosine_sim_api(text_vector, image_vector))
    cal_end = time.time()
    logger.info("Calculate similarities in {}s".format(cal_end-cal_start))


    logger.info("Start generate heatmap ...")
    generate_start = time.time()

    # read Image
    source_img = cv2.imread(img_path)
    img_row = np.shape(source_img)[0]
    img_col = np.shape(source_img)[1]

    # mkdir map
    heat_map = np.zeros([img_row, img_col], dtype=float)
    heat_num = np.zeros([img_row, img_col], dtype=float)
    for idx,file in enumerate(subimages):
        r_start, r_end, c_start, c_end = file.replace(".jpg","").split("_")
        heat_map[int(r_start):int(r_end), int(c_start):int(c_end)] += sim_results[idx]
        heat_num[int(r_start):int(r_end), int(c_start):int(c_end)] += 1
        
    for i in range(np.shape(heat_map)[0]):
        for j in range(np.shape(heat_map)[1]):
            heat_map[i,j] = heat_map[i,j] / heat_num[i,j]

    logger.info("Generate finished, start optim ...")
    # filter
    adaptive = np.asarray(heat_map)
    adaptive = adaptive-np.min(adaptive)
    heatmap = adaptive/np.max(adaptive)
    # must convert to type unit8
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.medianBlur(heatmap,251)
    img_add = cv2.addWeighted(source_img, 0.7, heatmap, 0.3, 0)
    generate_end = time.time()
    logger.info("Generate heatmap in {}s".format(generate_end-generate_start))

    # save
#    logger.info("Saving heatmap in {} ...".format(heatmap_dir))
#    cv2.imwrite(os.path.join(heatmap_dir, "heatmap.png"),heatmap)
#    cv2.imwrite(os.path.join(heatmap_dir, "heatmap_add.png"),img_add)

    logger.info("Saving heatmap in {} ...".format(output_file_h))
    logger.info("Saving heatmap in {} ...".format(output_file_a))
    cv2.imwrite( output_file_h ,heatmap)
    cv2.imwrite( output_file_a ,img_add)
    logger.info("Saved ok.")

    # clear temp
    utils.delete_dire(subimages_dir)
    os.rmdir(subimages_dir)

#    return  heatmap_dir


def semantic_localization(request_data):
    logger.info("Request json: {}".format(request_data))

    # 检测请求完备性
    if not isinstance(request_data, dict):
        return utils.get_stand_return(False, "Request must be dicts, and have keys: input_file, output_file, params.")
    if 'input_file' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: list input_file.")
    if 'output_file' not in request_data.keys():
        return utils.get_stand_return(False, "Request must have keys: list output_file.")
    if ('params' in request_data.keys()) and ('steps' in request_data['params'].keys()):
        steps = request_data['params']['steps']
    else:
        steps = [128,256,512]

    # 解析
    image_path, text, params, output_file_h, output_file_a = request_data['input_file'][0], request_data['params']['text'],  request_data['params'], request_data['output_file'][0], request_data['output_file'][1]

    # 判断文件格式
    if not (image_path.endswith('.tif') or image_path.endswith('.jpg') or image_path.endswith('.tiff') or image_path.endswith('.png')):
        return utils.get_stand_return(False, "File format is uncorrect: only support .tif, .tiff, .jpg, and .png .")
    else:
        split_image(image_path, steps)
        generate_heatmap(image_path, text, output_file_h, output_file_a)
        return utils.get_stand_return(True, "Generate successfully.")
