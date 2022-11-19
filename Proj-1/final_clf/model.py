import torch
import functools

@torch.no_grad()
def inspect_grad(grad): 
    assert torch.isnan(grad).sum() == 0, 'backward grad contains nan!'
    return grad

class ContrastClf(torch.nn.Module):
    def __init__(self) -> None:
        super(ContrastClf, self).__init__()
        self.transpose_0 = lambda x: x.transpose(-3, -1)
        self.conv_layer_0 = torch.nn.Sequential(
            # imgout=(..., 60, 32 - 0, 32 - 0)
            torch.nn.Conv2d(3, 60, (5, 5), stride=(1,1), padding=(2,2), groups=3),
            torch.nn.LeakyReLU(),
            # imgout=(..., 50, 32 - 2, 32 - 2)
            torch.nn.Conv2d(60, 50, (3, 3), stride=(1,1), padding=(0,0), groups=1),
            # imgout=(..., 40, 30 - 2, 30 - 2)
            torch.nn.Conv2d(50, 32, (5, 5), stride=(1,1), padding=(0,0), groups=1),
            # imgout=(..., 32, 28 - 4, 28 - 4)
            torch.nn.MaxPool2d((5, 5), stride=(1,1)),
            # imgout=(..., 50, 26 - 2, 26 - 2)
            torch.nn.Conv2d(32, 20, (3, 3), stride=(1,1), padding=(0,0), groups=1),
            # imgout=(..., 40, 30 - 2, 30 - 2)
            torch.nn.Conv2d(20, 16, (5, 5), stride=(1,1), padding=(0,0), groups=1),
            # imgout=(..., 32, 28 - 2, 28 - 2)
            torch.nn.MaxPool2d((5, 5), stride=(1,1)),
        )
        self.lin_layer_0 = torch.nn.Sequential(
            torch.nn.Flatten(-3, -1),
            torch.nn.Linear(16*12*12, 16*10*10),
            torch.nn.ReLU(),
            torch.nn.Linear(16*10*10, 8*10*10),
            torch.nn.ReLU(),
            torch.nn.Linear(8*10*10, 8*5*5),
            torch.nn.ReLU(),
            torch.nn.Linear(8*5*5, 10),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.transpose_0(x)
        x = self.conv_layer_0(x)
        x = self.lin_layer_0(x)
        return x

    def irregularity(self):
        return functools.reduce(lambda x, y: x+y.pow(2).sum(), self.parameters(), 0)

    def batch_loss(self, x, y):
        y_pred = self.forward(x)
        return -y_pred[torch.arange(y.shape[0]), y].log().mean()

    def norm(self, x):
        return torch.cdist(x, torch.zeros((1, x.shape[-1]), device=x.device))

    @torch.no_grad()
    def nan_to_zero(self):
        for x in self.parameters():
            if x.requires_grad: x.grad.nan_to_num_(0)

if __name__ == '__main__':
    clf = ContrastClf().to('cuda:0')
    x = torch.randint(255, size=(512, 32, 32, 3)).to(torch.float).to('cuda:0')
    y = torch.randint(10, size=(512, )).to(torch.long).to('cuda:0')
    print(clf.batch_loss(x, y))
    clf.irregularity()