# focus Attention

## 项目说明

- 本项目主要旨在解决现有模型对与南特704数据集中胚胎图像的七个焦平面利用不充分的情况
- 通过focus-attention学习七个焦平面的特征
- 用胚胎发育阶段预测任务判断模型性能

## 数据集说明

**存储位置：**`I:\datasets\胚胎\南特704`

#### 数据集组织形式

```bash
南特704/
	embryo_dataset/
		AA83-7/
			D2013.01.28_S0717_I132_WELL7_RUN1.jpeg
			D2013.01.28_S0717_I132_WELL7_RUN2.jpeg
			D2013.01.28_S0717_I132_WELL7_RUN3.jpeg
			......
			D2013.01.28_S0717_I132_WELL7_RUN285.jpeg
		AAL839-6/
		AB028-6/
		......
    embryo_dataset_F15/
    embryo_dataset_F30/
    embryo_dataset_F45/
    embryo_dataset_F-15/
    embryo_dataset_F-30/
    embryo_dataset_F-45/
```

## 代码说明

### 项目结构

```bash
南特704/
    checkpoints/
    configs/
    data/
    dataset/
    model/
    test/
    trainer/
    weights/
    train.py
```

### dataset结构

```bash
{
	"images": imgs				//[7x1x224x224]-[FxCxHxW]
    "embryo": embryo_name,		//ep.AA83-7
    "image_name": img.name,		//ep.D2013.01.28_S0717_I132_WELL7_RUN10.jpeg
    "run": run_id,				//ep.10
    "label": label				//ep.0
}
```

### 标签匹配

```bash
{
    "tPB2": 0,
    "tPNa": 1, 
    "tPNf": 2, 
    "t2": 3, 
    "t3": 4, 
    "t4": 5, 
    "t5": 6, 
    "t6": 7,
    "t7": 8, 
    "t8": 9, 
    "t9": 10, 
    "tM": 11, 
    "tSB": 12,
    "tB": 13, 
    "tEB": 14,
    "tHB": 15
}
```

