import torch
import psutil

CPU_CORE_PER_GPU = 8
DRAM_PER_GPU = 48
SYSTEM_DISK = 100
GPU_MEMORY = 6

def test_cpu_cores():
    num_cpu_core = psutil.cpu_count(logical=False)
    num_gpu = torch.cuda.device_count()
    assert num_cpu_core >= CPU_CORE_PER_GPU
    # assert more than 8 cpu core(physical) per gpu
    assert num_cpu_core / num_gpu >= 8
    
    
def test_memory():
    num_gpu = torch.cuda.device_count()
    memory = psutil.virtual_memory().total
    # estimate 48G MEM per card
    assert memory / num_gpu >= (DRAM_PER_GPU * 0.9 * 1000 * 1000 * 1000)
    
    
def test_disk():
    disk_space = psutil.disk_usage('/').total
    # estimate 100G disk '/'
    assert disk_space >= (SYSTEM_DISK * 1000 * 1000 * 1000)


def test_gpu_memory():
    count = torch.cuda.device_count()
    assert count >= 1
    for i in range(count):
        gpu_mem = torch.cuda.get_device_properties(i).total_memory
        # estimate 6G GPU MEM
        assert gpu_mem >= GPU_MEMORY * 1000 * 1000 * 1000
