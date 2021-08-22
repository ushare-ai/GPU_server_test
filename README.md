# GPU_server_test

测试共19项；

11项单元测试，测试CPU，内存，硬盘，显卡，显存，显卡欺骗器，cuda可用性等；

7项集成测试，测试实际各种使用情况是否正常（如pytorch计算 训练 推理）, 排除潜在硬件故障；

1项benchmark速度测试，测试数十种CNN网络的训练和推理，持续约十分钟；

无红色error则通过，有红色F / error会显示报错对应测试项，可根据信息排查; 

4卡2080ti全测试过程约10分钟，若测试时间过长如超过半小时，则机器可能存在问题可以提前中止测试（不中止则需要等测试完成后才会报error）;

结束后生成result文件夹导出性能报告;

--------------------------------------------------------------

依赖:
```
安装完驱动后继续安装python相关依赖（可不安装cuda）:
> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh ./
> chmod +x Miniconda3-latest-Linux-x86_64.sh
> sh Miniconda3-latest-Linux-x86_64.sh
> source ~/.bashrc
一路默认安装完conda，开始安装pytorch，如cuda11时:
> pip install numpy
> pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
> pip install pytest psutil pandas
然后cd到本目录下: 运行: 【pytest .】或单独运行某项测试【pytest 01_unit_test】/ 【pytest 02_integration_test】；
> cd path/to/GPU_server_test/ 
> pytest .
```
