# Incremental Shift OOD (IS-OOD) Benchmark

This repository is the PyTorch implementation of the IS-OOD benchmark mentioned in the paper [`Rethinking the Evaluation of Out-of-Distribution Detection: A Sorites Paradox`](???),
aiming to evaluate the performance of OOD detection models across different levels of semantic and covariate shifts.

For convenience, this repository adopts the OOD detection APIs provided by [OpenOOD](https://github.com/Jingkang50/OpenOOD). Therefore, any OOD detection method implemented based on OpenOOD API can be evaluated in this benchmark. Try evaluating your own work on the benchmark!

## Get Started

### Data and Checkpoint
Download all the data and the checkpoint pre-trained on ImageNet-1K, and place them in the corresponding directories:
 - ImageNet-1K and ImageNet-21K datasets can be downloaded from official website [here](https://image-net.org/download.php).
 - Syn-IS dataset can be obtained from our [Drive](???).
 - The checkpoint file can be downloaded from official website [here](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights).

Once you have prepared the data and the checkpoint, your directory should look like this:
```
ISOOD
├── openood
│   └── ...
├── data
│   ├── imglist
│   ├── imagenet1k
│   ├── imagenet21k
│   └── SynIS
├── pretrained_weight
│   └── resnet50_imagenet1k_v1.pth
├── calculate_metrics.py
├── analyze_metrics.py
├── ...
```

### Evaluation
Calculate the metrics on ImageNet-21K subsets and Syn-IS for a certain OOD detection method:
```
python calculate_metrics.py --postprocessor msp
```

Analyze the results in the saved csv file:
```
python analyze_metrics.py --postprocessor msp --metric AUROC
```

---


<!-- ## Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@article{long2024isood,
  year={2024}
}
``` -->

