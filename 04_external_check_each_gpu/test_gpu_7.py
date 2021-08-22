import torch
import torchvision.models as models

gpu_id = 7
count = torch.cuda.device_count()


def test_ex_torch_calculate_each_card():
    if gpu_id >= count:
        return True
    a = torch.randn(1024, 512).to("cuda:%d"%gpu_id)
    b = torch.randn(512, 1024).to("cuda:%d"%gpu_id)
    c = torch.matmul(a, b)
    d = torch.randn(1024, 1024).to("cuda:%d"%gpu_id)
    e = c + d * 0.25
    assert e.shape == torch.Size([1024, 1024])


def test_ex_gpu_memory_each_card():
    if gpu_id >= count:
        return True
    a = torch.randn(10000, 10000).to("cuda:%d"%gpu_id)
    b = torch.randn(10000, 10000).to("cuda:%d"%gpu_id)
    for _ in range(5):
        a = torch.matmul(a, b)
        c = torch.randn(10000, 10000).to("cuda:%d"%gpu_id)
        b = 0.25 * (c - a)


def test_ex_inference_each_card():
    if gpu_id >= count:
        return True
    a = torch.randn(16,3,224,224).to("cuda:%d"%gpu_id)
    model = models.resnet18(pretrained=False)
    model.to("cuda:%d"%gpu_id)
    out = model(a)
    assert out.shape == torch.Size([16, 1000])


def test_ex_training_each_card():
    if gpu_id >= count:
        return True
    a = torch.randn(2,3,224,224).to("cuda:%d"%gpu_id)
    b = torch.randint(1000,(2,)).to("cuda:%d"%gpu_id)
    model = models.resnet18(pretrained=False).to("cuda:%d"%gpu_id)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    optimizer.zero_grad()
    out = model(a)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(out, b)
    optimizer.step()