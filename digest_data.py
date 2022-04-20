import os
from PIL import Image
import pandas as pd


src = r'C:\Users\Aboud\Datasets\NUS_hand_posture_data_2\webcam10'
des = r'C:\Users\Aboud\Datasets\NUS_hand_posture_data_2\Data'
FLIP = True  # ALWAYS CHECK HERE TO NOT MESS THE DATA


def get_images(path):
    return [x for x in os.listdir(path) if x.endswith('.jpg')]


def get_next(lst):
    separated = [(x[0], x[3:-5]) for x in lst if x[3:-5].isdigit()]
    s = pd.DataFrame(separated)
    s[1] = s[1].astype(int)
    return dict(s.groupby(0).max().reset_index(drop=False).values.tolist())


def map_place(lst, dic):
    to = []
    for i in lst:
        dic[i[0]] += 1
        new = f'{i[0]} ({dic[i[0]]}).jpg'
        to.append(new)
    return lst, to


def digest(src_n, des_n, flip=False):
    for old, new in zip(src_n, des_n):
        old_path = os.path.join(src, old)
        im = Image.open(old_path).resize((160, 120))
        if not flip:
            out = im
            out_t = im.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            out = im.transpose(Image.FLIP_LEFT_RIGHT)
            out_t = im
        out.save(os.path.join(des, new))
        shift = chr(ord(new[0]) + 1) + new[1:]
        out_t.save(os.path.join(des, shift))


if __name__ == '__main__':
    scr_f = get_images(src)
    des_f = get_images(des)
    next_nums = get_next(des_f)
    src_names, des_names = map_place(scr_f, next_nums)
    digest(src_names, des_names, flip=FLIP)
