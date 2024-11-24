# Subjects200K Dataset


<a href="https://arxiv.org/abs/2411.xxxx"><img src="https://img.shields.io/badge/ariXv-2411.xxxx-A42C25.svg" alt="arXiv"></a>
<a href="https://github.com/Yuanshi9815/OminiControl"><img src="https://img.shields.io/badge/GitHub-OminiControl-blue.svg?logo=github&" alt="GitHub"></a>

<img src='./assets/data.jpg' width='100%' />
</br>
A large-scale dataset of 200,000 paired images (derived from OmniControl) where each pair maintains subject consistency while varying the scene context.

## Quick Start
```python
from src.dataset import Subjects200K

# Initialize dataset
dataset = Subjects200K()

# Access samples
sample = dataset[0]
```

### Sample Format
Each data point contains:
- `instance`: Brief description of the subject
- `image1`: Left image (512x512)
- `image2`: Right image (512x512)
- `description1`: Text description for left image
- `description2`: Text description for right image
- `image_pair`: Combined image (1024x512)

## Citation
```
@article{
  tan2024omini,
  title={OminiControl: Minimal and Universal Control for Diffusion Transformer},
  author={Zhenxiong Tan, Xingyi Yang, Songhua Liu, and Xinchao Wang},
  journal={arXiv preprint arXiv:2411.xxxx},
  year={2024}
}
```
