
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def load_models(args, model):
    
    if not args.ckpt_all == "":

        load = torch.load(args.ckpt_all)
        mis_keys, unexp_keys = model.load_state_dict(load, strict=False)
        print('missing_keys:', mis_keys)
        print('unexpected_keys:', unexp_keys)
    
    elif not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)


def MAE(input, target):
    return torch.abs(input - target).mean()

def MAPE(input, target):
    return torch.abs((input - target) / target).mean() * 100

def MSE(input, target):
    num = 1
    for s in input.size():
        num = num * s
    return (input - target).pow(2).sum().item() / num

def EdgeWiseKL(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask = (input > 0) & (target > 0)
    input = input.masked_select(mask)
    target = target.masked_select(mask)
    kl = (target * torch.log(target / input)).sum().item() / num
    return kl

def MissRate(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask1 = (input > 0) & (target == 0)
    mask2 = (input == 0) & (target > 0)
    mask = mask1 | mask2
    return mask.sum().item() / num


def set_seed(seed):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)