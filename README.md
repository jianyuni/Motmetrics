# Motmetrics

运行文件：apps/mainfile.py

在 command line 输入 python mainfile.py args.groundtruth args.detction 即可 

args.groundtruths 为gt的绝对路径 同样的 args.detection 为det绝对路径

该项目兼容MOT15-2D 文件格式
详细数据格式请阅读网站 https://arxiv.org/pdf/1603.00831.pdf 

MOT16 全部数据已经整合在data文档里
与原MOT Challenge 数据不同的是在增加了文件序列号在第一列并用逗号隔开，以便讲所有的gt和det各整合在一个文档。
