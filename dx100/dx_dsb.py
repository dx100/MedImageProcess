#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:44:30 2017

@author: dx100
"""


import os

import sys
import numpy as np
sys.path.append('/Users/dx100/Learn/Python/Kaggle/DSB2017/')
from preprocessing import full_prep
from dx_config_submit import config as config_submit

# Step 1, prepare data
datapath = config_submit['datapath']
prep_result_path = config_submit['preprocess_result_path']
skip_prep = config_submit['skip_preprocessing']
skip_detect = config_submit['skip_detect']

if not skip_prep:
    testsplit = full_prep(datapath,prep_result_path,
                          n_worker = config_submit['n_worker_preprocessing'],
                          use_existing=config_submit['use_exsiting_preprocessing'])
else:
     #DX changed the following to exclude .DS_Store
 #   testsplit = os.listdir(datapath)
    testsplit = [f for f in os.listdir(datapath) if not f.startswith('.')]
    
    
# DX: by now, the prep_results contains the clear image (in 3D)
# End of Step 1.

# DX: this is the detection, not the training case, so skip the step 2. 
# The step 2 is in ./training folder
    
# Step 3, detect a nodule
# DX: now, try to isolate out the nodule
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from importlib import import_module

from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier

from split_combine import SplitComb
from dx_test_detect import test_detect

nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
config1, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load(config_submit['detector_param'])
nod_net.load_state_dict(checkpoint['state_dict'])

#torch.cuda.set_device(0)
#nod_net = nod_net.cuda()
#cudnn.benchmark = True
nod_net = DataParallel(nod_net)

bbox_result_path = './bbox_result'
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)
#testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

if not skip_detect:
    margin = 32
    sidelen = 144
    config1['datadir'] = prep_result_path
    split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])

    dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
    test_loader = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers = 32,pin_memory=False,collate_fn =collate)
    
    #This is the slow one, about 30min per case.
    test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])

# End of Step 3, the nodule detection, the output is the pbb and lbb file in bbox directory.
    
# Step 4. Training a cancer classifier.
casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
casenet = casemodel.CaseNet(topk=5)
config2 = casemodel.config
checkpoint = torch.load(config_submit['classifier_param'])
casenet.load_state_dict(checkpoint['state_dict'])

#torch.cuda.set_device(0)
#casenet = casenet.cuda()
#cudnn.benchmark = True
casenet = DataParallel(casenet)

filename = config_submit['outputfile']

def test_casenet(model,testset):
    data_loader = DataLoader(
        testset,
        batch_size = 1,
        shuffle = False,
        num_workers = 32,
        pin_memory=False)
    #model = model.cuda()
    model.eval()
    predlist = []
    
    from torch.autograd import Variable    
    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i,(x,coord) in enumerate(data_loader):

        coord = Variable(coord)
        x = Variable(x)
        nodulePred,casePred,_ = model(x,coord)
        predlist.append(casePred.data.cpu().numpy())
        #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
    predlist = np.concatenate(predlist)
    return predlist    

config2['bboxpath'] = bbox_result_path
config2['datadir'] = prep_result_path

dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
predlist = test_casenet(casenet,dataset).T
# DX changed to the following in order to get the df
import pandas as pd
from collections import OrderedDict
anstable = OrderedDict(({'id':testsplit, 'cancer':predlist}))
df = pd.DataFrame(anstable)
df.to_csv(filename,index=False)
