import os


#dataset_path = '/media/anpei/win/shanghaiTech/faces/data/1024x1024/FFHQ/images'

dataset_path = '/mnt/data/new_disk/chenap/dataset/ffhq/images/'

#dataset_path = '/mnt/data/new_disk/chenap/dataset/CelebAMask-HQ/image'

ckpt_path = '/mnt/data/new_disk/chenap/code/SPADE-face/results/Jack12test/SPADE/'

img_size = 512

for i in range(2, 12, 2):

    #generatedata_path = "/media/anpei/win/shanghaiTech/code/pix2pixHD-face/results/Jack12test/label2img-pix2pixHD/test_{}/images".format(i)
    
    generatedata_path = os.path.join(ckpt_path, 'epoch_{}'.format(i))
    os.system("python fid_score.py {} {} --img_size {}".format(generatedata_path, dataset_path, img_size))