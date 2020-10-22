# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Preprocess data for CLEAR_MOT_M."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from configparser import ConfigParser
import logging
import time

import numpy as np

import motmetrics.distances as mmd
from motmetrics.lap import linear_sum_assignment


def preprocessResult(res, gt, seqLength):
    """Preprocesses data for utils.CLEAR_MOT_M.

    Returns a subset of the predictions.
    """
    # pylint: disable=too-many-locals
    st = time.time()
    labels = [
        'ped',               # 1
        'person_on_vhcl',    # 2
        'car',               # 3
        'bicycle',           # 4
        'mbike',             # 5
        'non_mot_vhcl',      # 6
        'static_person',     # 7
        'distractor',        # 8
        'occluder',          # 9
        'occluder_on_grnd',  # 10
        'occluder_full',     # 11
        'reflection',        # 12
        'crowd',             # 13
    ]
    distractors = ['person_on_vhcl', 'static_person', 'distractor', 'reflection']
    is_distractor = {i + 1: x in distractors for i, x in enumerate(labels)}
    for i in distractors:
        is_distractor[i] = 1
    todrop = []
    
    for t in range(1, seqLength + 1):
        if t not in res.index or t not in gt.index:
            continue
        resInFrame = res.loc[t]
        #print (gt.index)
        #print(resInFrame)
        GTInFrame = gt.loc[t]
        #print(gt.loc[t])
        A = GTInFrame[['X', 'Y', 'Width', 'Height']].values
        B = resInFrame[['X', 'Y', 'Width', 'Height']].values
        #print(type(A),type(B))
        disM = mmd.iou_matrix(A, B, max_iou=0.5)
        le, ri = linear_sum_assignment(disM)
        #print(disM)
        flags = [
            1 if is_distractor[it['ClassId']] or it['Visibility'] < 0. else 0
            for i, (k, it) in enumerate(GTInFrame.iterrows())
        ]
        hid = [k for k, it in resInFrame.iterrows()]
        '''
        for i in flags:
            print(i)
        '''
        for i, j in zip(le, ri):
            #print(1)
            if not np.isfinite(disM[i, j]):
                continue
            if flags[i]:
                #print('yes')
                todrop.append((t, hid[j]))
    #print(res)
    '''
    for i in todrop:
        print(i)
    '''
    if not (len(todrop)==0):
        ret = res.drop(labels=todrop)
    else:
        ret = res
    logging.info('Preprocess take %.3f seconds and remove %d boxes.',
                 time.time() - st, len(todrop))
    return ret
