<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

<div align="center"><a title="" href="https://github.com/zjykzj/multigrain-pytorch"><img align="center" src="./imgs/MultiGrain.png" alt=""></a></div>

<p align="center">
  «MultiGrain» re-implements the paper <a href="https://arxiv.org/abs/1902.05509">MultiGrain: a unified image embedding for classes and instances</a>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

实现逻辑：

1. 先完成分类任务训练
2. 然后扩展到分类+检索任务联合训练

如果仅仅进行分类任务训练，那么它是简单的。包括模型（ResNet50）、损失函数（CrossEntropyLoss）、优化器（SGD）的格式 和之前的方式差别不大

如果是加入检索任务训练，那么需要在数据采样、模型输出、损失函数计算、优化器更新上均进行了变化

先进行分类任务训练，基于ImageNet数据集，使用MultiGrain提供的预处理器、模型、损失函数以及优化器

然后加入检索任务训练

```shell
cd classification

export PYTHONPATH=.

python3 train.py --model resnet50 --lr 1e-2 --data-path /data/sdf/imagenet/ --output-dir ./outputs --ra-reps 1 --batch-size 128 --epochs 120
torchrun --nproc_per_node=8 train.py --model resnet50 --lr 1e-2 --data-path /data/sdf/imagenet/ --output-dir ./outputs --ra-reps 1 --batch-size 128 --epochs 120
```

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

[MultiGrain](https://arxiv.org/abs/1902.05509) provides a unified framework to simultaneously train classification and retrieval tasks. In addition, it also provides source code implementation - [facebookresearch/multigrain](https://github.com/facebookresearch/multigrain). This warehouse is modified on the original basis to deepen the understanding and use of the MultiGrain framework.

## Installation

...

## Usage

...

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [facebookresearch/multigrain](https://github.com/facebookresearch/multigrain)

```text
@ARTICLE{2019arXivMultiGrain,
       author = {Berman, Maxim and J{\'e}gou, Herv{\'e} and Vedaldi Andrea and
         Kokkinos, Iasonas and Douze, Matthijs},
        title = "{{MultiGrain}: a unified image embedding for classes and instances}",
      journal = {arXiv e-prints},
         year = "2019",
        month = "Feb",
}
```

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/multigrain-pytorch/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2022 zjykzj