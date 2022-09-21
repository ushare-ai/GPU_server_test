"""Compare speed of different models"""
import torch
import torchvision.models as models
import platform
import psutil
import torch.nn as nn
import datetime
import time
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True


class RandomDataset(Dataset):

    def __init__(self,  length):
        self.len = length
        self.data = torch.randn(3, 224, 224, length)

    def __getitem__(self, index):
        return self.data[:,:,:,index]

    def __len__(self):
        return self.len
    

MODEL_LIST = {
    models.resnet: models.resnet.__all__[1:],
    models.densenet: models.densenet.__all__[1:],
    models.vgg: models.vgg.__all__[1:],
}

precisions=["float"]

# Training settings

WARM_UP = 5
NUM_TEST = 50
BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_CLASSES = 1000
NUM_GPU = torch.cuda.device_count()
folder = 'result'

BATCH_SIZE*=NUM_GPU

folder_name=folder

rand_loader = DataLoader(dataset=RandomDataset(BATCH_SIZE*(WARM_UP + NUM_TEST)),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

def train(precision='single'):
    """use fake image for training speed test"""
    target = torch.LongTensor(BATCH_SIZE).random_(NUM_CLASSES).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)(pretrained=False)
            if NUM_GPU > 1:
                model = nn.DataParallel(model)
            model=getattr(model,precision)()
            model=model.to('cuda')
            durations = []
            print(f'Benchmarking Training {precision} precision type {model_name} ')
            for step,img in enumerate(rand_loader):
                img=getattr(img,precision)()
                torch.cuda.synchronize()
                start = time.time()
                model.zero_grad()
                prediction = model(img.to('cuda'))
                loss = criterion(prediction, target)
                loss.backward()
                torch.cuda.synchronize()
                end = time.time()
                if step >= WARM_UP:
                    durations.append((end - start)*1000)
            print(f'{model_name} model average train time : {sum(durations)/len(durations)}ms')
            del model
            benchmark[model_name] = durations
    return benchmark


def inference(precision='float'):
    benchmark = {}
    with torch.no_grad():
        for model_type in MODEL_LIST.keys():
            for model_name in MODEL_LIST[model_type]:
                model = getattr(model_type, model_name)(pretrained=False)
                if NUM_GPU > 1:
                    model = nn.DataParallel(model)
                model=getattr(model,precision)()
                model=model.to('cuda')
                model.eval()
                durations = []
                print(f'Benchmarking Inference {precision} precision type {model_name} ')
                for step,img in enumerate(rand_loader):
                    img=getattr(img,precision)()
                    torch.cuda.synchronize()
                    start = time.time()
                    model(img.to('cuda'))
                    torch.cuda.synchronize()
                    end = time.time()
                    if step >= WARM_UP:
                        durations.append((end - start)*1000)
                print(f'{model_name} model average inference time : {sum(durations)/len(durations)}ms')
                del model
                benchmark[model_name] = durations
    return benchmark


f"{platform.uname()}\n{psutil.cpu_freq()}\ncpu_count: {psutil.cpu_count()}\nmemory_available: {psutil.virtual_memory().available}"

def test_main():
    start = time.time()
    device_name=str(torch.cuda.get_device_name(0))
    device_name=f"{device_name}_{NUM_GPU}_gpus_"
    system_configs=f"{platform.uname()}\n\
                     {psutil.cpu_freq()}\n\
                    cpu_count: {psutil.cpu_count()}\n\
                    memory_available: {psutil.virtual_memory().available}"
    gpu_configs=[torch.cuda.device_count(),torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.get_device_name(0)]
    gpu_configs=list(map(str,gpu_configs))
    temp=['Number of GPUs on current device : ','CUDA Version : ','Cudnn Version : ','Device Name : ']

    os.makedirs(folder_name, exist_ok=True)
    now = datetime.datetime.now()
    
    start_time=now.strftime('%Y/%m/%d %H:%M:%S')
    
    print(f'benchmark start : {start_time}')

    for idx,value in enumerate(zip(temp,gpu_configs)):
        gpu_configs[idx]=''.join(value)
        print(gpu_configs[idx])
    print(system_configs)

    with open(os.path.join(folder_name,"system_info.txt"), "w") as f:
        f.writelines(f'benchmark start : {start_time}\n')
        f.writelines('system_configs\n\n')
        f.writelines(system_configs)
        f.writelines('\ngpu_configs\n\n')
        f.writelines(s + '\n' for s in gpu_configs )

    
    for precision in precisions:
        train_result=train(precision)
        train_result_df = pd.DataFrame(train_result)
        path=f'{folder_name}/{device_name}_{precision}_model_train_benchmark.csv'
        train_result_df.to_csv(path, index=False)

        inference_result=inference(precision)
        inference_result_df = pd.DataFrame(inference_result)
        path=f'{folder_name}/{device_name}_{precision}_model_inference_benchmark.csv'
        inference_result_df.to_csv(path, index=False)

    now = datetime.datetime.now()

    end_time=now.strftime('%Y/%m/%d %H:%M:%S')
    print(f'benchmark end : {end_time}')
    with open(os.path.join(folder_name,"system_info.txt"), "a") as f:
        f.writelines(f'benchmark end : {end_time}\n')

    end = time.time()
    time_costs = end - start
    assert time_costs < 1200
