import torchvision.models as models
from torch import nn
import torch
from torch.nn import functional as F
from dataset import *
from torch import optim
from torch.utils.data import DataLoader
import torch.jit as jit


class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)), requires_grad=True)
        self.func = nn.Softmax()

    def forward(self, x, s=1, m=0.2):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)

        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(
            s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))

        return arcsoftmax


class FaceNet(nn.Module):

    def __init__(self):
        super(FaceNet, self).__init__()
        self.sub_net = nn.Sequential(
            models.mobilenet_v2(pretrained=True),

        )
        # print( models.mobilenet_v2())
        self.feature_net = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, 512, bias=False),
        )
        self.arc_softmax = Arcsoftmax(512, 8)

    def forward(self, x):
        y = self.sub_net(x)
        feature = self.feature_net(y)
        return feature, self.arc_softmax(feature, 1, 1)

    def encode(self, x):
        return self.feature_net(self.sub_net(x))


def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    print(face1_norm.shape)
    print(face2_norm.shape)
    cosa = torch.matmul(face1_norm, face2_norm.T)
    return cosa


if __name__ == '__main__':
    # 训练过程
    # net = FaceNet().cuda()
    # loss_fn = nn.NLLLoss()
    # optimizer = optim.Adam(net.parameters())
    #
    # dataset = MyDataset("data3")
    # dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    #
    # for epoch in range(100000):
    #     for xs, ys in dataloader:
    #         feature, cls = net(xs.cuda())
    #
    #         loss = loss_fn(torch.log(cls), ys.cuda())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print(torch.argmax(cls, dim=1), ys)
    #     print(str(epoch)+"Loss====>"+str(loss.item()))
    #     if epoch%100==0:
    #         torch.save(net.state_dict(), "params/1.pt")
    #         print(str(epoch)+"参数保存成功")

    # 使用
    net = FaceNet().cuda()
    net.load_state_dict(torch.load("params/1.pt"))
    net.eval()

    person1 = tf(Image.open("test_img/pic0.jpg")).cuda()
    person1_feature = net.encode(torch.unsqueeze(person1, 0))
    # person1_feature = net.encode(person1[None, ...])
    print(person1.shape)
    print(torch.unsqueeze(person1, 0).shape)
    print(person1[None, ...].shape)

    person2 = tf(Image.open("test_img/pic10.jpg")).cuda()
    # person2 = tf(Image.open("test_img/1.bmp")).cuda()
    person2_feature = net.encode(person2[None, ...])

    siam = compare(person1_feature, person2_feature)
    print(siam)

    # 把模型和参数进行打包，以便C++或PYTHON调用
    # x = torch.Tensor(1, 3, 112, 112)
    # traced_script_module = jit.trace(net, x)
    # traced_script_module.save("model.cpt")
