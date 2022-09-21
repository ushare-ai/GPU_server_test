from torch.utils import benchmark
import pandas as pd
from collections import defaultdict
import inspect
import torch
import sys
import os

os.makedirs('result', exist_ok=True)
sys.stdout = open('result/flops_bandwidth_test.log', mode='w', encoding='utf-8')

print('Pytorch version\t:', torch.__version__)
print('CUDA version\t:', torch.version.cuda)
print('GPU\t\t:', torch.cuda.get_device_name())


pd.options.display.precision = 3


def var_dict(*args):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return dict([(name, val) for name, val in callers_local_vars if val is arg][0]
                for arg in args)


def walltime(stmt, arg_dict, duration=3):
    return benchmark.Timer(stmt=stmt, globals=arg_dict).blocked_autorange(
        min_run_time=duration).median

def test_tflops_bandwidth():
    with open('result/flops_bandwidth_test.log', 'w') as f:
        count = torch.cuda.device_count()
        for gpu_id in range(count):
            f.write('\n>>>>>>>>>>>>>>> [Testing GPU:%d] >>>>>>>>>>>>>>>\n'%gpu_id)
            f.write('Test TFLOPS\n')
            f.write('='*45 + '\n')
            matmul_tflops = defaultdict(lambda: {})
            for n in [128, 512, 2048, 8192]:
                for dtype in (torch.float32, torch.float16):
                    a = torch.randn(n, n, dtype=dtype).to("cuda:%d"%gpu_id)
                    b = torch.randn(n, n, dtype=dtype).to("cuda:%d"%gpu_id)
                    t = walltime('a @ b', var_dict(a, b))
                    matmul_tflops[f'n={n}'][dtype] = 2*n**3 / t / 1e12
                    del a, b
            f.write(str(pd.DataFrame(matmul_tflops)) + '\n')
            f.write('='*45 + '\n')

            f.write('\nTest Bandwidth\n')
            f.write('='*45 + '\n')
            vector = defaultdict(lambda: {})
            for n in [1024*64, 1024*256, 1024*1024, 1024*1024*4]:
                a = torch.randn(n).to("cuda:%d"%gpu_id)
                t = walltime('a * 1.2', var_dict(a))
                vector[n]['TFLOPS'] = n / t / 1e12
                vector[n]['GB/s'] = 8 * n / t / 1e9
            f.write(str(pd.DataFrame(vector)) + '\n')
            f.write('='*45 + '\n')
