# GPU_server_test

测试共15项；

6项单元测试，测试显存，cuda可用性等；

7项集成测试，测试实际各种使用情况是否正常（如pytorch计算 训练 推理）, 排除潜在硬件故障；

2项benchmark速度测试，测试矩阵运算的TFLOPS以及带宽，以及测试数十种CNN网络的训练和推理，持续约十分钟；

无红色error则通过，有红色F / error会显示报错对应测试项，可根据信息排查; 

2080ti全测试过程约10分钟，若测试时间过长如超过半小时，则机器可能存在问题可以提前中止测试（不中止则需要等测试完成后才会报error）;

结束后生成result文件夹导出性能报告;

--------------------------------------------------------------

使用方法：
```
pytest .
```
