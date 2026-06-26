# Frequency Integrated Transformer for  Single Image Arbitrary Scale Super Resolution
Xufei Wang, Fei Ge, Ling Zheng^, Shizhuang Weng^

Methods based on implicit neural representation have demonstrated remarkable capabilities in arbitrary-scale super-resolution (ASSR) tasks, but they neglect the potential value of the frequency domain, leading to sub-optimal performance. We propose a novel network called Frequency-Integrated Transformer (FIT) to incorporate and utilize frequency information to enhance ASSR performance. FIT employs a Frequency Incorporation Module (FIM) to introduce frequency information in a component-retentive manner and Frequency Utilization Self-Attention module(FUSAM) to eﬃciently leverage frequency information by exploiting spatial-frequency interrelationship and global nature of frequency. FIM enriches detail characterization by incorporating frequency information through a combination of Fast Fourier Transform (FFT) with realimaginary mapping. In FUSAM, Interaction Implicit Self-Attention (IISA) achieves cross-domain information synergy by interacting spatial and frequency information in subspace, while Frequency Correlation Self Attention (FCSA) captures the global context by computing correlation in frequency. Experimental results on multiple benchmark datasets show that FIT achieves the best average performance among the compared methods. The gains are also observed across a wide range of upsampling scales, including both in-scale and out-scale settings, indicating the stable scale-adaptive ability of FIT. Inaddition, visual feature maps, frequency error maps, and local attribution maps form a thoughtful diagnostic framework for interpreting FIT from feature, frequency and context perspectives.

> ^: corresponding author(s)
## Dependencies & Installation
```shell
conda create -n fit python
conda activate fit
pip install torch torchvision einops timm matplotlib
```

## Test
```shell
python test.py -config config.yaml -model model.pth -name name
```
## Training
```shell
python train.py -config config.yaml -name name
```

## Citation

If FIT helps your research or work, please consider citing the following works:

----------
```BibTex
@article{wang2025frequency,
  title={Frequency-Integrated Transformer for Arbitrary-Scale Super-Resolution},
  author={Wang, Xufei and Ge, Fei and Zhu, Jinchen and Zhang, Mingjian and Wu, Qi and Weng, Jifeng Ren Shizhuang},
  journal={arXiv preprint arXiv:2504.18818},
  year={2025}
}
```
