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

```shell
cd multigrain
export PYTHONPATH=.

python3 train.py --model resnet50 --lr 1e-2 --data-path /data/imagenet/ --output-dir ./outputs --ra-reps 1 --batch-size 128 --epochs 120
torchrun --nproc_per_node=8 train.py --model resnet50 --lr 1e-2 --data-path /data/imagenet/ --output-dir ./outputs --ra-reps 1 --batch-size 128 --epochs 120
```

使用DistributedSampler

```shell
torchrun --nproc_per_node=8 train.py --model resnet50 --lr 0.2 --data-path /data/imagenet/ --output-dir ./outputs --batch-size 256 --epochs 120 --classify-weight 0.5 --pooling-exponent 3 --ra-reps 3 --amp --lr-warmup-epochs 5 --lr-warmup-method linear
```

```text
Epoch: [117]
Acc@1 75.816 Acc@5 92.600
```

使用RASampler

```shell
torchrun --nproc_per_node=8 train.py --model resnet50 --lr 0.2 --data-path /data/imagenet/ --output-dir ./outputs --batch-size 256 --epochs 120 --classify-weight 0.5 --pooling-exponent 3 --ra-sampler --ra-reps 3 --amp --lr-warmup-epochs 5 --lr-warmup-method linear
```

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [facebookresearch/multigrain](https://github.com/facebookresearch/multigrain)
* [pytorch/vision](https://github.com/pytorch/vision)

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