import os

dataset_path = '/media/anpei/win/shanghaiTech/faces/data/1024x1024/FFHQ/images'

for i in range(2, 16, 2):

    generatedata_path = "/media/anpei/win/shanghaiTech/code/pix2pixHD-face/results/Jack12test/label2img-pix2pixHD/test_{}/images".format(i)

    os.system("python fid_score.py {} {}".format(generatedata_path, dataset_path))