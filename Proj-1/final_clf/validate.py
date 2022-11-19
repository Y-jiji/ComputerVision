from model import *
from data_util import *
import torch
from tqdm import trange

# configuration
device = torch.device('cuda:0')
candidate_k = [7,25,100,1000]
config_set = dict()

# ckpt dir
dir_path = os.path.dirname(__file__)

def validate(timestamp, path='final_clf/ckpt'):
    with torch.no_grad():
        print('load data')
        x_trn, y_trn, x_tst, y_tst = load_cifar_tensor(None, None, device)
        # x_trn = x_trn.to(torch.float)
        # y_trn = y_trn.to(torch.long)
        x_tst = x_tst.to(torch.float)
        y_tst = y_tst.to(torch.long)
        print('load model')
        model = ContrastClf().to(device)
        model.load_state_dict(torch.load(f'{path}/{timestamp}.model'))
        print('acc: ', ((model.forward(x_tst[:2000, ...]).argmax(-1) == y_tst[:2000]) * 1.0).mean())

if __name__ == '__main__':
    validate('202210161847')