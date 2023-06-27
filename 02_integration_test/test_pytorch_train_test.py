import torch
import torchvision.models as models


def test_inference_each_card():
    count = torch.cuda.device_count()
    for i in range(count):
        a = torch.randn(16,3,224,224).to("cuda:%d"%i)
        model = models.resnet18(pretrained=False)
        model.to("cuda:%d"%i)
        out = model(a)
        assert out.shape == torch.Size([16, 1000])


def test_inference_dp():
    a = torch.randn(16,3,224,224).cuda()
    model = models.resnet18(pretrained=False).cuda()
    model = torch.nn.DataParallel(model)
    out = model(a)
    assert out.shape == torch.Size([16, 1000])


def test_training_each_card():
    count = torch.cuda.device_count()
    for i in range(count):
        a = torch.randn(2,3,224,224).to("cuda:%d"%i)
        b = torch.randint(1000,(2,)).to("cuda:%d"%i)
        model = models.resnet18(pretrained=False).to("cuda:%d"%i)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
        optimizer.zero_grad()
        out = model(a)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(out, b)
        loss.backward()
        optimizer.step()


def test_training_dp():
    a = torch.randn(16,3,224,224).cuda()
    b = torch.randint(1000,(16,)).cuda()
    model = models.resnet18(pretrained=False).cuda()
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    optimizer.zero_grad()
    out = model(a)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(out, b)
    loss.backward()
    optimizer.step()


def test_training_dp():
    a = torch.randn(16,3,224,224).cuda()
    b = torch.randint(1000,(16,)).cuda()
    model = models.resnet18(pretrained=False).cuda()
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    optimizer.zero_grad()
    out = model(a)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(out, b)
    loss.backward()
    optimizer.step()
