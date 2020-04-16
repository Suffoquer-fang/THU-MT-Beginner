# from utils import *
from model import RNNLM, train, test, build_model, predict, test_perp
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm
# from data_loader import Dictionary, Corpus
# from torch import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


if __name__ == "__main__":
    a = torch.LongTensor([1, 2, 3]).to(device)
    print(a.dim())
    
    
    model = build_model()

    # # train(model, 'LDC-zh', 'log')

    model.load_state_dict(torch.load('LDC_result/model-1.ckpt'))
    model.eval()

    test_perp(model, '../../LDC/devtest/nist02/nist02.en0')
    test_perp(model, '../../LDC/devtest/nist02/nist02.en1')
    test_perp(model, '../../LDC/devtest/nist02/nist02.en2')
    test_perp(model, '../../LDC/devtest/nist02/nist02.en3')
    test_perp(model, '../../LDC/devtest/nist03/nist03.en0')
    test_perp(model, '../../LDC/devtest/nist04/nist04.en0')
    test_perp(model, '../../LDC/devtest/nist05/nist05.en0')
    test_perp(model, '../../LDC/devtest/nist06/nist06.en0')
    test_perp(model, '../../LDC/devtest/nist08/nist08.en0')
    # test_perp(model, '../../LDC/train/train.shuf.en')
    # train(model, 'LDC', 'log-1', False)
    







