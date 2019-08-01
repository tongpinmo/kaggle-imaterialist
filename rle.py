from itertools import groupby
from pycocotools import mask as mutils
import numpy as np
from tqdm import tqdm

def kaggle_rle_decode(rle, h, w):
    '''
    rle: run-length as string format(start length)
    shape: (height, width) of array to return
    returns numpy array, 1 - mask, 0 - background
    '''
    s = rle.split()
    starts, lenghts = [np.asarray(x, dtype = int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lenghts
    img = np.zeros(w*h, dtype = np.uint8)
    for _lo,_hi in zip(starts,ends):
        img[_lo:_hi]=1


    return  img.reshape((w, h)).T

def kaggle_rle_encode(mask):
    '''
    mask: numpy array, 1 - mask, 0 - background
    returns run length as string formated
    '''
    pixels = mask.ravel(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    rle = np.where(pixels[1:]!= pixels[:-1])[0] + 1
    rle[1:: 2] -= rle[::2]

    return ' '.join(str(x) for x in rle)

def coco_rle_encode(mask):
    rle = {
        'counts':[],
        'size': list(mask.shape)
    }

    counts = rle.get('counts')

    for i, (value, elements) in enumerate(groupby(mask.ravel(order = 'F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))


    return rle

def coco_rle_decode(rle, h, w):
    return mutils.decode(mutils.frPyObjects(rle, h ,w))

def kaggle2coco(kaggle_rle, h, w):
    if not len(kaggle_rle):
        return {'counts': [h * w],'size':[h,w]}

    roll2 = np.roll(kaggle_rle, 2)
    roll2[:2] = 1

    roll1 = np.roll(kaggle_rle, 1)
    roll1[:1] = 0

    if h*w != kaggle_rle[-1] + kaggle_rle[-2] - 1:
        shift = 1
        end_value = h*w - kaggle_rle[-1] -kaggle_rle[-2] + 1
    else:
        shift = 0
        end_value = 0

    coco_rle = np.full(len(kaggle_rle) + shift, end_value)
    coco_rle[:len(coco_rle) - shift] = kaggle_rle.copy()
    coco_rle[: len(coco_rle) - shift:2] = (kaggle_rle - roll1 - roll2)[::2].copy()
    return {'counts': coco_rle.tolist(), 'size':[h, w]}

if __name__ == '__main__':
    # debug
    for _ in tqdm(range(30)):
        h = np.random.randint(1,1000)
        w = np.random.randint(1,1000)
        mask = np.random.randint(0, 2, h*w).reshape(h,w)

        kaggle_rle = kaggle_rle_encode(mask)
        coco_rle = coco_rle_encode(mask)

        assert coco_rle == kaggle2coco(kaggle_rle, h, w)
        assert np.all(mask == kaggle_rle_decode(kaggle_rle,h,w))
        assert np.all (mask == coco_rle_decode(coco_rle, h,w))