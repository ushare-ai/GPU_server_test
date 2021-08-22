import torch
import torchvision.models as models


def test_pytorch_cuda_available():
    available = torch.cuda.is_available()
    print('CUDA available:', available)


def test_gpu_basic_info():
    count = torch.cuda.device_count()
    cards = []
    for i in range(count):
        cards.append(torch.cuda.get_device_name("cuda:%d"%i))
        device = torch.device("cuda:%d"%i)
    info = {'num_gpus': count, 'cards_name': cards}
    print(info)


def test_pytorch_cuda():
    a = torch.randn(1024, 1024)
    a.cuda()


def test_each_cuda_devices():
    count = torch.cuda.device_count()
    for i in range(count):
        a = torch.randn(1024, 1024)
        a.cuda(i)


def test_resnet18_each_devices():
    count = torch.cuda.device_count()
    for i in range(count):
        resnet18 = models.resnet18(pretrained=False)
        resnet18.to("cuda:%d"%i)


def test_calculate_each_devices():
    count = torch.cuda.device_count()
    for i in range(count):
        a = torch.randn(1024, 512).to("cuda:%d"%i)
        b = torch.randn(512, 1024).to("cuda:%d"%i)
        c = torch.matmul(a, b)
        d = torch.randn(1024, 1024).to("cuda:%d"%i)
        e = c + d * 0.25
        assert e.shape == torch.Size([1024, 1024])
