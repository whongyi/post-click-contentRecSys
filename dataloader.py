import numpy as np


def loadSpotify():
    data = {}
    data['train'] = np.load('./dataset/spotify/train.npy')
    data['val'] = np.load('./dataset/spotify/val.npy')
    data['test'] = np.load('./dataset/spotify/test.npy')
    data['pos_test'] = np.load('./dataset/spotify/pos_test.npy')
    data['neg_test'] = np.load('./dataset/spotify/neg_test.npy')
    data['total_users'] = 229792
    data['total_items'] = 100586
    return data

    
def loadByteDance():
    data = {}
    data['train'] = np.load('./dataset/bytedance/train.npy')
    data['val'] = np.load('./dataset/bytedance/val.npy')
    data['test'] = np.load('./dataset/bytedance/test.npy')
    data['pos_test'] = np.load('./dataset/bytedance/pos_test.npy')
    data['neg_test'] = np.load('./dataset/bytedance/neg_test.npy')
    data['total_users'] = 37043
    data['total_items'] = 271259
    return data
