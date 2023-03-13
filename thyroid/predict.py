import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np
from skimage.measure import label as sklabel
from skimage.measure import regionprops
from skimage.transform import resize
from torchvision import transforms as T
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd
import segmentation_models_pytorch as smp
import math
import shutil
import sys
import zlib


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    drawpoly(img,pts,color,thickness,style)

def contours_in(img, contours, h, w):
    p = np.zeros(shape=(h, w))
    cv2.drawContours(p, [contours], -1, 255, -1)
    a = np.where(p == 255)[0].reshape(-1, 1)
    b = np.where(p == 255)[1].reshape(-1, 1)
    coordinate = np.concatenate([a, b], axis=1).tolist()
    inside = [tuple(x) for x in coordinate]
    coords = np.nonzero(p)
    r = np.mean(img[coords])
    return inside, r


def contours_round(contours, h, w):
    p = np.zeros(shape=(h, w))
    cv2.drawContours(p, [contours], -1, 255, 1)
    a = np.where(p == 255)[0].reshape(-1, 1)
    b = np.where(p == 255)[1].reshape(-1, 1)
    coordinate = np.concatenate([a, b], axis=1).tolist()
    round = [tuple(x) for x in coordinate]
    return round


def near_ponits(round):
    near = []
    for point in round:
        near.append((point[0], point[1]))
        for i in range(8):
            for j in range(8):
                near.append((point[0] - i, point[1] - j))
                near.append((point[0] + i, point[1] + j))
    # print(near)
    return near

def largestConnectComponent(bw_img):
    if np.sum(bw_img) == 0:
        return bw_img
    # labeled_img, num = sklabel(bw_img, neighbors=4, background=0, return_num=True)
    labeled_img, num = sklabel(bw_img, background=0, return_num=True)
    if num == 1:
        return bw_img

    max_label = 0
    max_num = 0
    for i in range(0, num):
        # print(i)
        if np.sum(labeled_img == (i + 1)) > max_num:
            max_num = np.sum(labeled_img == (i + 1))
            max_label = i + 1
    mcr = (labeled_img == max_label)
    return mcr.astype(np.int)


def preprocess(mask_c1_array_biggest, c1_size=256):
    if np.sum(mask_c1_array_biggest) == 0:
        minr, minc, maxr, maxc = [0, 0, c1_size, c1_size]
    else:
        region = regionprops(mask_c1_array_biggest)[0]
        minr, minc, maxr, maxc = region.bbox

    dim1_center, dim2_center = [(maxr + minr) // 2, (maxc + minc) // 2]
    max_length = max(maxr - minr, maxc - minc)

    max_lengthl = int((c1_size / 256) * 80)
    preprocess1 = int((c1_size / 256) * 19)
    pp22 = int((c1_size / 256) * 31)

    if max_length > max_lengthl:
        ex_pixel = preprocess1 + max_length // 2
    else:
        ex_pixel = pp22 + max_length // 2

    dim1_cut_min = dim1_center - ex_pixel
    dim1_cut_max = dim1_center + ex_pixel
    dim2_cut_min = dim2_center - ex_pixel
    dim2_cut_max = dim2_center + ex_pixel

    if dim1_cut_min < 0:
        dim1_cut_min = 0
    if dim2_cut_min < 0:
        dim2_cut_min = 0
    if dim1_cut_max > c1_size:
        dim1_cut_max = c1_size
    if dim2_cut_max > c1_size:
        dim2_cut_max = c1_size
    return [dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max]


def TNSCUI_preprocess(image_path, outputsize=256, orimg=False):
    # image_path = r'/media/root/老王3号/challenge/tnscui2020_train/image/1273.PNG'
    img = Image.open(image_path)
    Transform = T.Compose([T.ToTensor()])
    img_tensor = Transform(img)
    img_dtype = img_tensor.dtype
    img_array_fromtensor = (torch.squeeze(img_tensor)).data.cpu().numpy()

    img_array = np.array(img, dtype=np.float32)

    or_shape = img_array.shape  # 原始图片的尺寸

    value_x = np.mean(img, 1)  # % 为了去除多余行，即每一列平均
    value_y = np.mean(img, 0)  # % 为了去除多余列，即每一行平均

    x_hold_range = list((len(value_x) * np.array([0.24 / 3, 2.2 / 3])).astype(np.int))
    y_hold_range = list((len(value_y) * np.array([0.8 / 3, 1.8 / 3])).astype(np.int))
    # x_hold_range = list((len(value_x) * np.array([0.8 / 3, 2.2 / 3])).astype(np.int))
    # y_hold_range = list((len(value_y) * np.array([0.8 / 3, 2.2 / 3])).astype(np.int))

    # value_thresold = 0
    value_thresold = 5

    x_cut = np.argwhere((value_x <= value_thresold) == True)
    x_cut_min = list(x_cut[x_cut <= x_hold_range[0]])
    if x_cut_min:
        x_cut_min = max(x_cut_min)
    else:
        x_cut_min = 0

    x_cut_max = list(x_cut[x_cut >= x_hold_range[1]])
    if x_cut_max:
        # print('q')
        x_cut_max = min(x_cut_max)
    else:
        x_cut_max = or_shape[0]

    y_cut = np.argwhere((value_y <= value_thresold) == True)
    y_cut_min = list(y_cut[y_cut <= y_hold_range[0]])
    if y_cut_min:
        y_cut_min = max(y_cut_min)
    else:
        y_cut_min = 0

    y_cut_max = list(y_cut[y_cut >= y_hold_range[1]])
    if y_cut_max:
        # print('q')
        y_cut_max = min(y_cut_max)
    else:
        y_cut_max = or_shape[1]

    if orimg:
        x_cut_max = or_shape[0]
        x_cut_min = 0
        y_cut_max = or_shape[1]
        y_cut_min = 0
    # 截取图像
    cut_image = img_array_fromtensor[x_cut_min:x_cut_max, y_cut_min:y_cut_max]
    cut_image_orshape = cut_image.shape

    cut_image = resize(cut_image, (outputsize, outputsize), order=3)

    cut_image_tensor = torch.tensor(data=cut_image, dtype=img_dtype)

    return [cut_image_tensor, cut_image_orshape, or_shape, [x_cut_min, x_cut_max, y_cut_min, y_cut_max]]


def get_filelist_frompath(filepath, expname, sample_id=None):
    """
    读取文件夹中带有固定扩展名的文件
    :param filepath:
    :param expname: 扩展名，如'h5','PNG'
    :param sample_id: 可以只读取固定患者id的图片
    :return: 文件路径list
    """
    file_name = os.listdir(filepath)
    file_List = []
    if sample_id is not None:
        for file in file_name:
            if file.endswith('.' + expname):
                id = int(file.split('.')[0])
                if id in sample_id:
                    file_List.append(os.path.join(filepath, file))
    else:
        for file in file_name:
            if file.endswith('.' + expname):
                file_List.append(os.path.join(filepath, file))
    return file_List


def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    # print ("图像的高宽分别为：height,width",height,width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    # print("max_gray_level:",max_gray_level)
    return max_gray_level + 1


def getGlcm(input1, d_x, d_y):
    srcdata = input1.copy()
    gray_level = 16
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input1.shape

    max_gray_level = maxGrayLevel(input1)
    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def feature_computer(p):
    # con:对比度反应了图像的清晰度和纹理的沟纹深浅。纹理越清晰反差越大对比度也就越大。
    # eng:熵（Entropy, ENT）度量了图像包含信息量的随机性，表现了图像的复杂程度。当共生矩阵中所有值均相等或者像素值表现出最大的随机性时，熵最大。
    # agm:角二阶矩（能量），图像灰度分布均匀程度和纹理粗细的度量。当图像纹理均一规则时，能量值较大；反之灰度共生矩阵的元素值相近，能量值较小。
    # idm:反差分矩阵又称逆方差，反映了纹理的清晰程度和规则程度，纹理清晰、规律性较强、易于描述的，值较大。
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    gray_level = 16
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm


def predictor(sdir, csv_path, model_path):
    # read in the csv file
    class_df = pd.read_csv(csv_path)
    class_count = len(class_df['class'].unique())
    img_height = int(class_df['height'].iloc[0])
    img_width = int(class_df['width'].iloc[0])
    img_size = (img_width, img_height)
    scale = class_df['scale by'].iloc[0]
    image_list = []
    # determine value to scale image pixels by
    try:
        s = int(scale)
        s2 = 1
        s1 = 0
    except:
        split = scale.split('-')
        s1 = float(split[1])
        s2 = float(split[0].split('*')[1])
    model = load_model(model_path)
    image_list = []
    asm_list = []
    con_list = []
    eng_list = []
    idm_list = []
    file_list = []
    good_image_count = 0
    img = cv2.imread(sdir)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm_0 = getGlcm(img_gray, 1, 0)
    asm, con, eng, idm = feature_computer(glcm_0)
    asm_list.append(asm)
    con_list.append(con)
    eng_list.append(eng)
    idm_list.append(idm)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    good_image_count += 1
    img = img * s2 - s1
    image_list.append(img)
    file_name = os.path.split(sdir)[1]
    file_list.append(file_name)

    image_array = np.array(image_list)
    asm_array = np.array(asm_list)
    con_array = np.array(con_list)
    eng_array = np.array(eng_list)
    idm_array = np.array(idm_list)
    # make predictions on images, sum the probabilities of each class then find class index with
    # highest probability
    preds = model.predict([image_array, asm_array, con_array, eng_array, idm_array])
    psum = []
    for i in range(class_count):  # create all 0 values list
        psum.append(0)
    for p in preds:  # iterate over all predictions
        for i in range(class_count):
            psum[i] = psum[i] + p[i]  # sum the probabilities
    index = np.argmax(psum)  # find the class index with the highest probability sum
    klass = class_df['class'].iloc[index]  # get the class name that corresponds to the index
    prob = psum[index] / good_image_count  # get the probability average
    return klass, prob, img, None

def makeROI(img_path, tmp_dir):
    weight_c1 = r'D:\data\weigh_and_id\TNSCUI-1\fold1_s1.pkl'
    weight_c2 = r'D:\data\weigh_and_id\TNSCUI-1\fold1_s2.pkl'

    # 构建两个模型
    gray_path = tmp_dir + '/' + 'gray.jpg'
    original_path = tmp_dir + '/' + 'original.jpg'
    mask_path = tmp_dir + '/' + 'mask.jpg'
    contour_path = tmp_dir + '/' + 'contour.jpg'
    roi_path = tmp_dir + '/' + 'roi.jpg'
    rec_path = tmp_dir + '/' + 'rec.jpg'
    ellipse_path = tmp_dir + '/' + 'ellipse.jpg'
    foci_path = tmp_dir + '/' + 'foci.jpg'

    image = cv2.imread(img_path)
    img_gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(gray_path, img_gray)
    #cv2.imwrite(original_path, image)
    shutil.copy(img_path, original_path)

    file_list = []
    file_list.append(gray_path)
    with torch.no_grad():
        # 构建模型
        # cascade1
        model_cascade1 = smp.DeepLabV3Plus(encoder_name="efficientnet-b6", encoder_weights=None, in_channels=1,
                                                   classes=1)
        # model_cascade1.to(device)
        model_cascade1.load_state_dict(torch.load(weight_c1, map_location='cpu'))
        model_cascade1.eval()
        # cascade2
        model_cascade2 = smp.DeepLabV3Plus(encoder_name="efficientnet-b6", encoder_weights=None, in_channels=1,
                                                   classes=1)
        # model_cascade2.to(device)
        model_cascade2.load_state_dict(torch.load(weight_c2, map_location='cpu'))
        model_cascade2.eval()

        for index, img_file in enumerate(file_list):
            # 处理原图
            with torch.no_grad():
                img, cut_image_orshape, or_shape, location = TNSCUI_preprocess(img_file, outputsize=256, orimg=False)
                img = torch.unsqueeze(img, 0)
                img = torch.unsqueeze(img, 0)
                # img = img.to(device)
                img_array = (torch.squeeze(img)).data.cpu().numpy()
                """过一遍cascade1"""
                # 获取cascade1输出
                with torch.no_grad():
                    mask_c1 = model_cascade1(img)
                    mask_c1 = torch.sigmoid(mask_c1)
                    mask_c1_array = (torch.squeeze(mask_c1)).data.cpu().numpy()
                    mask_c1_array = (mask_c1_array > 0.5)
                    mask_c1_array = mask_c1_array.astype(np.float32)
                    # 获取最大联通域
                    mask_c1_array_biggest = largestConnectComponent(mask_c1_array.astype(np.int))

                """过一遍cascade2"""
                with torch.no_grad():
                    # 获取roi的bounding box坐标
                    dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max = preprocess(mask_c1_array_biggest, 256)
                    # 根据roi的bounding box坐标，获取img区域
                    img_array_roi = img_array[dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max]
                    img_array_roi_shape = img_array_roi.shape
                    img_array_roi = resize(img_array_roi, (512, 512), order=3)
                    img_array_roi_tensor = torch.tensor(data=img_array_roi, dtype=img.dtype)
                    img_array_roi_tensor = torch.unsqueeze(img_array_roi_tensor, 0)
                    img_array_roi_tensor = torch.unsqueeze(img_array_roi_tensor, 0)
                    # 获取cascade2输出,并还原大小
                    #print('use cascade2')
                    mask_c2 = model_cascade2(img_array_roi_tensor)
                    mask_c2 = torch.sigmoid(mask_c2)
                    mask_c2_array = (torch.squeeze(mask_c2)).data.cpu().numpy()
                    cascade2_t = 0.5
                    mask_c2_array = (mask_c2_array > cascade2_t)
                    mask_c2_array = mask_c2_array.astype(np.float32)
                    mask_c2_array = resize(mask_c2_array, img_array_roi_shape, order=0)
                    # 放回cascade1输出的mask
                    mask_c1_array_biggest = mask_c1_array_biggest.astype(np.float32)
                    mask_c1_array_biggest[dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max] = mask_c2_array
                    mask_c1_array_biggest = mask_c1_array_biggest.astype(np.float32)
                    #print('c2')

                # 根据预处理信息，首先还原到原始size，之后放回原图位置
                mask_c1_array_biggest = mask_c1_array_biggest.astype(np.float32)
                final_mask = np.zeros(shape=or_shape, dtype=mask_c1_array_biggest.dtype)
                mask_c1_array_biggest = resize(mask_c1_array_biggest, cut_image_orshape, order=1)
                final_mask[location[0]:location[1], location[2]:location[3]] = mask_c1_array_biggest

                # 变成二值图
                final_mask = (final_mask > 0.5)
                final_mask = final_mask.astype(np.float32)
                final_mask = final_mask * 255
                final_mask = final_mask.astype(np.uint8)
                # print('final')
                im = Image.fromarray(final_mask)
    im.save(mask_path)


    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # h, w = mask_2d.shape
    ret, thresh = cv2.threshold(mask_2d, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for i, contour in enumerate(contours):
        cnt = np.array(contour)
        area = cv2.contourArea(contour=contour, oriented=False)
        length = cv2.arcLength(curve=contour, closed=True)
        if (length == 0.0):
            continue
        if (area > max_area):
            max_area = area
            cnt1 = contour

    cv2.drawContours(image, [cnt1], 0, (0, 0, 255), 2)
    # 保存图像
    cv2.imwrite(contour_path[:-4] + ".jpg", image)
    return contour_path

def predictResult(tmp_dir):
    gray_path = tmp_dir + '/' + 'gray.jpg'
    mask_path = tmp_dir + '/' + 'mask.jpg'
    roi_path = tmp_dir + '/' + 'roi.jpg'
    if not os.path.exists(mask_path):
        return "error", "error"
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img = cv2.imread(gray_path)
    x, y, w, h = cv2.boundingRect(mask)
    h1, w1 = img.shape[0], img.shape[1]
    if (y - 10) < 0 or (x - 10) < 0:
        y0, x0 = y, x
    else:
        y0, x0 = y - 10, x - 10
    if (y + h + 10) > h1 or (x + w + 10) > w1:
        y1, x1 = y + h, x + w
    else:
        y1, x1 = y + h + 10, x + w + 10
    img_crop = img[y0:y1, x0:x1]
    cv2.imwrite(roi_path, img_crop)

    #csv_path = 'class_dict_100.csv'
    csv_path = r'D:\train_work\class_dict_100.csv'
    #model_path = 'EfficientNetB6-thyroid-86.97.h5'
    model_path = r'D:\train_work\EfficientNetB6-thyroid-86.97.h5'
    klass, prob, img, df = predictor(roi_path, csv_path, model_path)  # run the classifier
    # msg = f' image is of class {klass} with a probability of {prob * 100: 6.2f} %'
    if (klass == 0):
        msg1 = "良性"
        msg2 = "不需FNA，可保持6-12个月的随访间隔"
    else:
        msg1 = "恶性"
        msg2 = "建议FNA，必要时需采取手术治疗，并在术后进行长期随访"

    return msg1, msg2

def calRoundness(tmp_dir):
    mask_path = tmp_dir + '/' + 'mask.jpg'
    contour_path = tmp_dir + '/' + 'contour.jpg'
    ellipse_path = tmp_dir + '/' + 'ellipse.jpg'
    if not os.path.exists(mask_path):
        return "error"
    else:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, heriachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        for i, contour in enumerate(contours):
            cnt = np.array(contour)
            area = cv2.contourArea(contour=contour, oriented=False)
            length = cv2.arcLength(curve=contour, closed=True)
            if (length == 0.0):
                # print(name)
                continue
            if (area > max_area):
                max_area = area
                rect = cv2.minAreaRect(cnt)
                length1 = length
                cnt1 = contour
        round = (4 * math.pi * max_area) / (length1 * length1)

        img = cv2.imread(contour_path)
        ellipse = cv2.fitEllipse(cnt1)
        cv2.ellipse(img, ellipse, (255, 255, 0), 2)
        cv2.imwrite(ellipse_path, img)
        msg = f'结节的圆形度为{round}'
        return msg

def calAt(tmp_dir):
    original_path = tmp_dir + '/' + 'original.jpg'
    mask_path = tmp_dir + '/' + 'mask.jpg'
    rec_path = tmp_dir + '/' + 'rec.jpg'
    if not os.path.exists(mask_path):
        return "error"
    else:
        mask = cv2.imread(mask_path)
        im = cv2.imread(original_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #ret, thresh = cv2.threshold(mask, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #contours, heriachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, heriachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour=contour, oriented=False)
            length = cv2.arcLength(curve=contour, closed=True)
            if (length == 0.0):
                continue
            if (area > max_area):
                max_area = area
                cnt = contour
        x, y, w, h = cv2.boundingRect(cnt)
        at = h / w
        x_mid_left = x
        y_mid_left = y + h / 2
        x_mid_right = x + w
        y_mid_right = y + h / 2
        x_mid_up = x + w / 2
        y_mid_up = y
        x_mid_down = x + w / 2
        y_mid_down = y + h
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ellipse = cv2.fitEllipse(cnt)
        #cv2.ellipse(im, ellipse, (255, 255, 0), 2)

        s = (x_mid_left, y_mid_left)
        e = (x_mid_right, y_mid_right)
        drawrect(im, s, e, (0, 255, 255), 2, 'dotted')
        s1 = (x_mid_up, y_mid_up)
        e1 = (x_mid_down, y_mid_down)
        drawrect(im, s1, e1, (0, 255, 255), 2, 'dotted')
        cv2.imwrite(rec_path, im)
        msg = f'结节的纵横比为{at}'
        return msg

def showEchoFoci(tmp_dir):
    original_path = tmp_dir + '/' + 'original.jpg'
    mask_path = tmp_dir + '/' + 'mask.jpg'
    foci_path = tmp_dir + '/' + 'foci.jpg'
    if not os.path.exists(mask_path):
        return "error"
    else:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img = cv2.imread(original_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h0, w0, _ = img.shape
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        x, y, w, h = cv2.boundingRect(thresh)
        contours, heriachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        num = 0
        num_in = 0
        for i, contour in enumerate(contours):
            cnt = np.array(contour)
            area = cv2.contourArea(contour=contour, oriented=False)
            length = cv2.arcLength(curve=contour, closed=True)
            if (length == 0.0):
                continue
            if (area > max_area):
                max_area = area
                cnt1 = contour

        points, r = contours_in(img_gray, cnt1, h0, w0)
        coords = contours_round(cnt1, h0, w0)
        near = near_ponits(coords)
        for point in points:
            num += 1
            i, j = point[0], point[1]
            if (img_gray[i][j] >= r + 35 and point not in near):
                img[i][j][0], img[i][j][1], img[i][j][0] = [0, 255, 0]
                num_in += 1
        rate = num_in / num
        msg = f'结节的钙化比例为{rate}'

        cv2.drawContours(img, [cnt1], 0, (0, 0, 255), 2)
        cv2.imwrite(foci_path, img)

        return msg


def crc32_hash(filename):
    with open(filename, 'rb') as f:
        return hex(zlib.crc32(f.read()) & 0xffffffff)[2:].zfill(8)