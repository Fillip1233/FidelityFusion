import sys

import torch
import numpy as np

from MFGP import *
from MFGP.utils.normalizer import Dateset_normalize_manager

def get_testing_data(fidelity_num):
    x = np.load(r'assets/sample_data/input.npy')
    y_list = [np.load(r'assets/sample_data/output_fidelity_{}.npy'.format(i)) for i in range(3)]
    y_list = y_list[:fidelity_num]

    x = torch.tensor(x)
    y_list = [torch.tensor(_) for _ in y_list]

    sample_num = x.shape[0]
    tr_x = x[:sample_num//2, ...].float()
    eval_x = x[sample_num//2:, ...].float()
    tr_y_list = [y[:sample_num//2, ...].float() for y in y_list]
    eval_y_list = [y[sample_num//2:, ...].float() for y in y_list]

    return tr_x, eval_x, tr_y_list, eval_y_list
    
def normalize_data(tr_x, eval_x, tr_y_list, eval_y_list):
    # normalize
    norm_tool = Dateset_normalize_manager([tr_x], tr_y_list)
    tr_x = norm_tool.normalize_input(tr_x, 0)
    tr_y_list = norm_tool.normalize_outputs(tr_y_list)
    eval_x = norm_tool.normalize_input(eval_x, 0)
    eval_y_list = norm_tool.normalize_outputs(eval_y_list)

    return tr_x, eval_x, tr_y_list, eval_y_list, norm_tool


model_dict = {
    'AR': AR_MODULE,
    'CIGAR': CIGAR_MODULE,
    'GAR': GAR_MODULE,
    'FIDES': MF_FIDES_MODULE,
    'NAR': NAR_MODULE,
    'ResGP': ResGP_MODULE,
    'CIGP': CIGP_MODULE,
    'HOGP': HOGP_MODULE,
}


if __name__ == '__main__':
    support_model = list(model_dict.keys())
    if len(sys.argv) < 2:
        print('Usage: python mfgp_demo.py <model_name>')
        print('support model: {}'.format(support_model))
        exit()
    elif sys.argv[1] not in support_model:
        print('model_name must be one of {}'.format(support_model))
        print('Got {}'.format(sys.argv[1]))
        exit()

    model_name = sys.argv[1]

    fidelity_num = 1 if model_name in ['CIGP', 'HOGP'] else 3
    tr_x, eval_x, tr_y_list, eval_y_list = get_testing_data(fidelity_num)
    
    src_y_shape = tr_y_list[0].shape[1:]
    if model_name in ['AR', 'CIGAR', 'FIDES', 'NAR', 'ResGP', 'CIGP']:
        flatten_output = True
        sample_num = tr_y_list[0].shape[0]
        tr_y_list = [_.reshape(sample_num, -1) for _ in tr_y_list]
        eval_y_list = [_.reshape(sample_num, -1) for _ in eval_y_list]

    # normalize data
    tr_x, eval_x, tr_y_list, eval_y_list, norm_tool = normalize_data(tr_x, eval_x, tr_y_list, eval_y_list)

    # init model
    config = {
        'fidelity_shapes': [_y.shape[1:] for _y in tr_y_list],
    }
    model_define = model_dict[model_name]
    model = model_define(config)

    # print info
    print('model: {}'.format(model_name))
    print('fidelity num: {}'.format(fidelity_num))
    print('x shape: {}'.format(tr_x.shape))
    print('y shape: {}'.format([_.shape for _ in tr_y_list]))

    # enable to test cuda
    if True and torch.cuda.is_available():
        print('enable cuda')
        model = model.cuda()
        tr_x = tr_x.cuda()
        eval_x = eval_x.cuda()
        tr_y_list = [_.cuda() for _ in tr_y_list]
        eval_y_list = [_.cuda() for _ in eval_y_list]

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    max_epoch = 300 if model_name in ['CIGAR', 'GAR'] else 50
        
    train_each_fidelity_separately = False
    if train_each_fidelity_separately and model_name not in ['CIGP', 'HOGP']:
        '''
            Train method 1: train each fidelity separately
        '''
        for _fn in range(fidelity_num):
            for epoch in range(max_epoch):
                optimizer.zero_grad()
                if _fn == 0:
                    low_fidelity = None
                else:
                    low_fidelity = tr_y_list[_fn-1]
                high_fidelity = tr_y_list[_fn]
                nll = model.single_fidelity_compute_loss(tr_x, low_fidelity, high_fidelity, fidelity_index=_fn)
                print('fidelity {}, epoch {}/{}, nll: {}'.format(_fn, epoch+1, max_epoch, nll.item()), end='\r')
                nll.backward()
                optimizer.step()
            print('\n')
    else:
        '''
            Train method 2: train all fidelity at the same time
        '''
        for epoch in range(max_epoch):
            optimizer.zero_grad()
            nll = model.compute_loss(tr_x, tr_y_list)
            nll.backward()
            optimizer.step()
            print('epoch {}/{}, nll: {}'.format(epoch+1, max_epoch, nll.item()), end='\r')


    # predict and plot result
    with torch.no_grad():
        predict_y = model(eval_x)[0]

    from MFGP.utils.plot_field import plot_container
    groudtruth = norm_tool.denormalize_output(eval_y_list[-1], fidelity_num-1)
    predict_y = norm_tool.denormalize_output(predict_y, fidelity_num-1)
    plot_container([groudtruth.reshape(-1, *src_y_shape), predict_y.reshape(-1, *src_y_shape)], 
                   ['ground truth', 'predict'], 0).plot(3)

