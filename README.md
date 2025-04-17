# Multi-Scale Implicit Transformer with Re-parameterize  for Arbitrary-Scale Super-Resolution
Xufei Wang, Fei Ge, Ling Zheng^, Shizhuang Weng^

Methods based on implicit neural representation have demonstrated remarkable capabilities in arbitrary-scale super-resolution (ASSR) tasks, but they neglect the potential value of the frequency domain, leading to suboptimal performance. We proposes a novel network called Frequency-Integrated Transformer (FIT) to incorporate and utilize frequency information to enhance performance.
	FIT employs Frequency Incorporation Module (FIM) to introduce frequency information in a lossless manner and Frequency-utilization Self-attention module (FUSAM) to efficiently leverage frequency information by exploiting spatial-frequency interrelationship and global nature of frequency.
	FIM enriches detail characterization by incorporating frequency information through a combination of Fast Fourier Transform (FFT) with real-imaginary mapping. In FUSAM, Interaction Implicit Self-Attention (IISA) achieves cross-domain information synergy by interacting spatial and frequency information in subspace, while Frequency Correlation Self-attention (FCSA) captures the global context by computing correlation in frequency.
	Experimental results demonstrate FIT yields superior performance compared to existing methods across multiple benchmark datasets.	Visual feature map proves the superiority of FIM over existing modules for enriching detail characterization. Frequency error map demonstrates sufficient coordination of cross-domain information by IISA. Local attribution map validates that FCSA effectively captures global context.  

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

If MSIT helps your research or work, please consider citing the following works:

----------
```BibTex
@article{wang2025fit,
  title={Frequency-Integrated Transformer for Arbitrary-Scale Super-Resolution},
  author={Wang, Xufei and Ge, Fei and Zheng, Ling and Weng, Shizhuang},
  journal={},
  year={2025}
}
```
