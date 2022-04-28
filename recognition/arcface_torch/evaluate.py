from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
import cv2
import datetime
import pickle
import mxnet as mx
from mxnet import ndarray as nd

import torch
import torchvision.transforms as T
from backbones import get_model



class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    # L2 distance
    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)

    # Cosine distance
    dist = np.dot(embeddings1, embeddings2.T).diagonal()

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        #print('threshold', thresholds[best_threshold_index])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,
                 threshold_idx], fprs[fold_idx,
                                      threshold_idx], _ = calculate_accuracy(
                                          threshold, dist[test_set],
                                          actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    best_threshold_acc = thresholds[best_threshold_index]
    return tpr, fpr, accuracy, best_threshold_acc


def calculate_accuracy(threshold, dist, actual_issame):
    # predict_issame = np.less(dist, threshold)
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    #print(true_accept, false_accept)
    #print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    # thresholds = np.arange(0, 4, 0.01)
    # thresholds = np.arange(0.2, 1, 0.01)
    thresholds = [0.6]
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_threshold_acc = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds)
    # thresholds = np.arange(0, 1, 1)
    # val, val_std, far = calculate_val(thresholds,
    #                                   embeddings1,
    #                                   embeddings2,
    #                                   np.asarray(actual_issame),
    #                                   1e-3,
    #                                   nrof_folds=nrof_folds)
    val = val_std = far = 0
    return tpr, fpr, accuracy, val, val_std, far, best_threshold_acc


def load_bin_torch(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  #py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  #py3

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     half = True
    # else:
    #     device = torch.device("cpu")

    data_list = []
    for flip in [0, 1]:
        data = torch.empty(
            (len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        b_img = np.asarray(bytearray(_bin))
        img = cv2.imdecode(b_img, cv2.IMREAD_COLOR)
        if img.shape[0] != image_size[0]:
            img = cv2.resize(img, image_size, cv2.INTER_CUBIC)
        for flip in [0, 1]:
            if flip == 1:
                img = cv2.flip(img, 1)
            last_img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125            
            
            last_img_torch = torch.from_numpy(last_img)
            # last_img_torch = last_img_torch.half() if half else last_img_torch.float()

            data_list[flip][i][:] = last_img_torch
        if i % 1000 == 0:
            print('loading bin', i)
    
    return (data_list, issame_list)


def test_torch(data_set,
              torch_model,
              batch_size,
              nfolds=10):
    
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    model = torch_model
    embeddings_list = []

    time_consumed = 0.0

    _label = np.ones((batch_size, ))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        half = True
    else:
        device = torch.device("cpu")

    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
        
            _data = data[ba:ba+count, ...]
            #print(_data.shape, _label.shape)
            time0 = datetime.datetime.now()

            # db = mx.io.DataBatch(data=(_data, ), label=(_label, ))
            
            db = _data.to(device)
            db = db.half() if half else db
            with torch.no_grad():
                net_out = model(db)
                _embeddings = net_out

            time_now = datetime.datetime.now()

            diff = time_now - time0

            time_consumed += diff.total_seconds()
            
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            try: 
                embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :].detach().cpu().numpy()
            except:
                embeddings[ba:bb, :] = _embeddings[:count, :].detach().cpu().numpy()
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    # for embed in embeddings_list:
    #     for i in range(embed.shape[0]):
    #         _em = embed[i]
    #         _norm = np.linalg.norm(_em)
    #         #print(_em.shape, _norm)
    #         _xnorm += _norm
    #         _xnorm_cnt += 1
    # _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)

    _, _, accuracy, val, val_std, far, best_thr_acc = evaluate(embeddings,
                                                 issame_list,
                                                 nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list, best_thr_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data-dir', default='', help='')
    parser.add_argument('--target',
                        default='lfw,cfp_ff,cfp_fp,agedb_30',
                        help='test targets.')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='./weights/backbone.pth')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--max', default='', type=str, help='')
    parser.add_argument('--mode', default=0, type=int, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')
    args = parser.parse_args()
    #sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
    #import face_image
    #prop = face_image.load_property(args.data_dir)
    #image_size = prop.image_size
    image_size = [112, 112]
    print('image_size', image_size)

    ctx = mx.gpu(args.gpu)
    nets = []
    vec = args.model.split(',')
    prefix = args.model.split(',')[0]
    epochs = []
    if len(vec) == 1:
        pdir = os.path.dirname(prefix)
        for fname in os.listdir(pdir):
            if not fname.endswith('.params'):
                continue
            _file = os.path.join(pdir, fname)
            if _file.startswith(prefix):
                epoch = int(fname.split('.')[0].split('-')[1])
                epochs.append(epoch)
        epochs = sorted(epochs, reverse=True)
        if len(args.max) > 0:
            _max = [int(x) for x in args.max.split(',')]
            assert len(_max) == 2
            if len(epochs) > _max[1]:
                epochs = epochs[_max[0]:_max[1]]

    else:
        epochs = [int(x) for x in vec[1].split('|')]
    print('model number', len(epochs))
    time0 = datetime.datetime.now()
    for epoch in epochs:
        # Pytorch model
        half = False
        if torch.cuda.is_available():
            device = torch.device("cuda")
            half = True
        else:
            device = torch.device("cpu")

        # statedict_path = 'training_mode/semi-siamese_training/logs/LOG_0604_RESNET50_IRSE_SemiSiamese_FP16/Epoch_50.pt'
        statedict_path = 'training_mode/semi-siamese_training/logs/LOG_2104_RESNET50_IRSE_SemiSiamese_FP16_2/Epoch_60.pt'
        model_name = 'r100'
        weight = './weights/backbone.pth'
        model = get_model(model_name, fp16=False)
        state_dict = torch.load(weight, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)

        if half:
            model.half()

        nets.append(model)

    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        path = os.path.join(args.data_dir, name + ".bin")
        if os.path.exists(path):
            print('loading.. ', name)
            # data_set = load_bin_np(path, image_size)
            data_set = load_bin_torch(path, image_size)

            ver_list.append(data_set)
            ver_name_list.append(name)

    for i in range(len(ver_list)):
        results = []
        for model in nets:
            # acc1, std1, acc2, std2, xnorm, embeddings_list, best_thr_acc = test_mx(
            #     ver_list[i], model, args.batch_size, args.nfolds)
            # acc1, std1, acc2, std2, xnorm, embeddings_list, best_thr_acc = test_onnx(
            #     ver_list[i], model, args.batch_size, args.nfolds)
            acc1, std1, acc2, std2, xnorm, embeddings_list, best_thr_acc = test_torch(
                ver_list[i], model, args.batch_size, args.nfolds)
            print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
            print('[%s]Accuracy: %1.5f+-%1.5f' %
                    (ver_name_list[i], acc1, std1))
            print('[%s]Accuracy-Flip: %1.5f+-%1.5f' %
                    (ver_name_list[i], acc2, std2))
            print(f'[{ver_name_list[i]}]Mean threshold: {best_thr_acc}')
            results.append(acc2)
        print('Max of [%s] is %1.5f' % (ver_name_list[i], np.max(results)))