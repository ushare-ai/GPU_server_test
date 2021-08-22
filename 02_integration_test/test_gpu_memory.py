import torch
import torchvision.models as models


def test_memory_each_devices():
    count = torch.cuda.device_count()
    for i in range(count):
        a = torch.randn(10000, 10000).to("cuda:%d"%i)
        b = torch.randn(10000, 10000).to("cuda:%d"%i)
        for _ in range(2):
            a = torch.matmul(a, b)
            c = torch.randn(10000, 10000).to("cuda:%d"%i)
            b = 0.25 * (c - a)
            

def test_cnn_dp_in_loop():
    model = models.resnet18(pretrained=False).cuda()
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)
    for _ in range(10):
        a = torch.randn(64,3,224,224).cuda()
        b = torch.randint(1000,(64,)).cuda()
        optimizer.zero_grad()
        out = model(a)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(out, b)
        optimizer.step()
