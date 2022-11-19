from time import localtime, strftime
from random import randint, random
from model import *          # model
from data_util import *      # load data
import os                    # path operation, etc. 
from multiprocessing import Process, Queue
import sys

# a simple configuration
sample_count = 1<<17
device = torch.device('cuda:0')
x_trn, y_trn, x_tst, y_tst = load_cifar_tensor(None, None, device)

# ckpt path
dir_path = os.path.dirname(__file__)

# a visualizer process
class RunningLoss(Process):
    def __init__(self, winnum, winprp, winlen, decay, queue) -> None:
        super(RunningLoss, self).__init__()
        self.window = [[]] * winnum
        self.winprp = winprp
        for prop in self.winprp:
            assert('color' in prop)
            assert('plot' in prop)
            assert('label' in prop)
        self.winlen = winlen
        self.decay  = decay
        self.queue  = queue

    def run(self):
        import matplotlib.pyplot as plt
        import time
        from random import random
        plt.ion()
        rand_throw_rate = 0.0
        while True:
            start_time = time.time_ns()
            item = self.queue.get()
            if random() < rand_throw_rate:
                rand_throw_rate *= 0.9
                continue
            get_time = time.time_ns()
            for index in range(len(self.window)):
                if len(self.window[index]) == 0:
                    self.window[index].append(item[index])
                else:
                    self.window[index].append(item[index]*(1-self.decay) + self.window[index][-1] * self.decay)
                if len(self.window[index]) > self.winlen:
                    self.window[index] = self.window[index][1:]
            plt.clf()
            for index in range(len(self.window)):
                prop = self.winprp[index]
                plt.subplot(*prop['plot'])
                plt.plot(
                    tuple(range(len(self.window[index]))), self.window[index], 
                    color=prop['color'], label=prop['label'])
                plt.legend()
            plt.draw()
            plt.pause(0.1)
            end_time = time.time_ns()
            rand_throw_rate = (end_time - get_time) / (end_time - start_time)

    def put(self, x):
        self.queue.put(x)

if __name__ == '__main__':
    # train script
    print('load model')
    model = ContrastClf().to('cuda:0')
    if len(sys.argv) >= 2:
        time_stamp = sys.argv[1]
        model.load_state_dict(torch.load(f'final_clf/ckpt/{time_stamp}.model'))
    else: 
        time_stamp = strftime(r"%Y%m%d%H%M", localtime())
        print(time_stamp)
    print('load data')
    trn_image, trn_label, tst_image, tst_label = load_cifar_tensor(None, None, 'cuda:0')
    print('initialize optimizer')
    batch_shape = (128, )
    optim = torch.optim.RMSprop(model.parameters(), lr=1e-7, momentum=0.9)
    epoch_size = 100000
    print('start training')
    try:
        loss_avg = None
        def update(x, y):
            if x is None: return y
            else: return 0.99 * x + 0.01 * y
        time = None
        for i, (x, label) in enumerate(sample(trn_image, trn_label, batch_shape, epoch_size)):
            model.zero_grad()
            loss = model.batch_loss(x, label)
            (loss + 0.05*model.irregularity()).backward()
            optim.step()
            loss_avg = update(loss_avg, loss.item())
            if i % 100 == 0:
                print(f'[batch {i}/{epoch_size}] {strftime(r"%Y%m%d%H%M", localtime())}')
                print(f':::: loss {loss_avg:.4f}')
                # parameter grad
                print('-'*20 + 'grad inspect' + '-'*20)
                for x in model.parameters():
                    if randint(1, 10) == 1:
                        print(round(x.grad.abs().mean().item(), 4))
                print('-'*(40+len('grad inspect')))
                print('')
    except KeyboardInterrupt:
        pass
    print('save model')
    torch.save(model.state_dict(), f'final_clf/ckpt/{time_stamp}.model')
