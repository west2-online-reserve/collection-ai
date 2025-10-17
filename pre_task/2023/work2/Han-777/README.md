* ResNet: 一开始用ResNet18和Resnet34训练发现准确率最高卡在0.6。最后采用了迁移学习的办法，固定预训练模型的参数，只训练最深的那层。在进行了15个epoch之后，训练集和测试集的识别准确率都能到0.6左右。此时将Resnet18中线性层与卷积层的参数开放，在原模型基础上继续训练7个epoch后测试集到达0.77左右的正确率。

* GRU: 采用ResNet-fc-GRU-Attention-fc 的结构，最后的测试集正确率在0.53左右，不是很理想。后续会继续改进。

  ![db69a08d11fa66eaffd6e7fd447b6f0](D:\Downloads\WeChat\WeChat Files\wxid_eio2sv5wh1di22\FileStorage\Temp\db69a08d11fa66eaffd6e7fd447b6f0.jpg)

