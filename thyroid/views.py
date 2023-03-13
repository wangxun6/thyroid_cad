from django.shortcuts import render
from django.http import HttpResponse
from thyroid_project import settings
from .predict import makeROI, predictResult, calRoundness, calAt, showEchoFoci, crc32_hash
import json
import os
import shutil
from .models import ThyroidImage
import uuid

# Create your views here.
def index(request):
    response = render(request, 'index.html')
    response['Cache-Control'] = 'no-cache'
    response['Expires'] = 0
    return response

def showroi(request):
    if request.method == "POST":
        f1 = request.FILES.get('file')
        # 用于识别
        fname = './static/img/%s' % f1.name
        tmp_dir = request.POST.get('path')
        with open(fname, 'wb') as pic:
            for c in f1.chunks():
                pic.write(c)
        file_path = makeROI(fname, tmp_dir)
        return HttpResponse(file_path)

def predict_class1(request):
    if request.method == "POST":
        tmp_dir = request.POST.get('path')
        original_path = tmp_dir + '/' + 'original.jpg'
        if os.path.exists(original_path):
            crc_value = crc32_hash(original_path)
            is_exists = ThyroidImage.objects.filter(crc32=crc_value).exists()
            if is_exists == True:
                cls = ThyroidImage.objects.get(crc32=crc_value).label
                if cls == 1:
                    msg1 = "恶性"
                    msg2 = "建议FNA，必要时需采取手术治疗，并在术后进行长期随访"
                else:
                    msg1 = "良性"
                    msg2 = "不需FNA，可保持6-12个月的随访间隔"
            else:
                msg1, msg2 = predictResult(tmp_dir)
        else:
            msg1, msg2 = "error", "error"
        res_dict = {"class":msg1,"advise":msg2}
        res_dict = json.dumps(res_dict)
        return HttpResponse(res_dict)


def cal_roundness(request):
    if request.method == "POST":
        tmp_dir = request.POST.get('path')
        msg1 = calRoundness(tmp_dir)
        res_dict = {"round":msg1}
        res_dict = json.dumps(res_dict)
        return HttpResponse(res_dict)

def cal_at(request):
    if request.method == "POST":
        tmp_dir = request.POST.get('path')
        msg1 = calAt(tmp_dir)
        res_dict = {"at":msg1}
        res_dict = json.dumps(res_dict)
        return HttpResponse(res_dict)


def cal_foci(request):
    if request.method == "POST":
        tmp_dir = request.POST.get('path')
        msg1 = showEchoFoci(tmp_dir)
        res_dict = {"foci":msg1}
        res_dict = json.dumps(res_dict)
        return HttpResponse(res_dict)


def clean_tmp(request):
    if request.method == "POST":
        path = './static/tmp'
        uuid0 = str(uuid.uuid1())
        # for file in os.listdir(path):
        #     file_data = os.path.join(path, file)
        #     if os.path.isfile(file_data) == True:
        #         os.remove(file_data)
        tmp_path = path + '/' + uuid0
        os.mkdir(tmp_path)
        return HttpResponse(tmp_path)


def save_modify(request):
    if request.method == "POST":
        save_path = './static/img/save'
        path1 = './static/img/save/1'
        path0 = './static/img/save/0'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.mkdir(path1)
            os.mkdir(path0)
        picname = request.POST.get('name')
        cls = request.POST.get('class')
        flag = request.POST.get('is_correct')
        if cls == "" or picname == "":
            return HttpResponse('error')
        else:
            if flag == "true":
                if (cls == '良性'):
                    dir1 = path0
                    label = 0
                elif (cls == '恶性'):
                    dir1 = path1
                    label = 1
            else:
                if (cls == '良性'):
                    dir1 = path1
                    label = 1
                elif (cls == '恶性'):
                    dir1 = path0
                    label = 0
            file_path = './static/img/%s' % picname
            crc_value = crc32_hash(file_path)
            name1 = picname.split('.')[0]
            affix = picname.split('.')[1]
            final_path = dir1 + '/' + name1 + '_' + crc_value + '.' + affix
            shutil.copy(file_path, final_path)
            is_exists = ThyroidImage.objects.filter(crc32=crc_value).exists()
            if is_exists != True:
                my_object = ThyroidImage(path=final_path, label=label, crc32=crc_value)
                my_object.save()
            return HttpResponse('success')