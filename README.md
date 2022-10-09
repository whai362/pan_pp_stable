## Introduction
This repository is the official implementation of [PAN++](https://arxiv.org/abs/2105.00405).
Compared to [pan_pp.pytorch](https://github.com/whai362/pan_pp.pytorch), this repository is specific to [PAN++](https://arxiv.org/abs/2105.00405), and more stable.

## Installation

First, clone the repository locally:

```shell
git clone https://github.com/whai362/pan_pp_stable.git
```

Then, install PyTorch 1.1.0+, torchvision 0.3.0+, and other requirements:

```shell
conda install pytorch torchvision -c pytorch
pip install -r requirement.txt
```

Finally, compile codes of post-processing:

```shell
# build pa and other post-processing algorithms
sh ./compile.sh
```

## Dataset
Please refer to [dataset/README.md](dataset/README.md) for dataset preparation.

## Training & Testing
ICDAR2015: please refer to [IC15_RESULTS.md](IC15_RESULTS.md) for training and testing.

RCTW-17: please refer to [RCTW17_RESULTS.md](RCTW17_RESULTS.md) for training and testing.

Total-Text: please refer to [TT_RESULTS.md](TT_RESULTS.md) for training and testing.

CTW1500: please refer to [CTW_RESULTS.md](CTW_RESULTS.md) for training and testing.

MSRA-TD500: please refer to [MSRA_RESULTS.md](MSRA_RESULTS.md) for training and testing.

### Evaluate the performance

```shell
cd eval/
./eval_{DATASET}.sh
```

### Evaluate the speed

```shell script
python test.py XXX --report_speed true
```


### Visualization

```shell script
python test.py XXX --vis true
```


## Citation

Please cite the related works in your publications if it helps your research:


### PAN++

```
@article{wang2021pan++,
  title={PAN++: Towards Efficient and Accurate End-to-End Spotting of Arbitrarily-Shaped Text},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Liu, Xuebo and Liang, Ding and Zhibo, Yang and Lu, Tong and Shen, Chunhua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```

## License

This project is developed and maintained by [IMAGINE Lab@National Key Laboratory for Novel Software Technology, Nanjing University](https://cs.nju.edu.cn/lutong/ImagineLab.html).

<img src="logo.jpg" alt="IMAGINE Lab">

This project is released under the [Apache 2.0 license](https://github.com/whai362/pan_pp_stable/blob/master/LICENSE).
