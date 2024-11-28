# Subjects200K Dataset


<a href="https://arxiv.org/abs/2411.15098"><img src="https://img.shields.io/badge/ariXv-2411.15098-A42C25.svg" alt="arXiv"></a>
<a href="https://github.com/Yuanshi9815/OminiControl"><img src="https://img.shields.io/badge/GitHub-OminiControl-blue.svg?logo=github&" alt="GitHub"></a>
<a href="https://huggingface.co/datasets/Yuanshi/Subjects200K"><img src="https://img.shields.io/badge/🤗_HuggingFace-Data-ffbd45.svg" alt="HuggingFace"></a>

<img src='./assets/data.jpg' width='100%' />
</br>

Subjects200K is a large-scale dataset containing 200,000 paired images, introduced as part of the [OminiControl](https://github.com/Yuanshi9815/OminiControl) project. Each image pair maintains subject consistency while presenting variations in scene context.

## Quick Start
### From HuggingFace (Recommended 🌟)
<a href="https://huggingface.co/datasets/Yuanshi/Subjects200K"><img src="https://img.shields.io/badge/🤗_HuggingFace-Data-ffbd45.svg" alt="HuggingFace"></a>

- Usage
  ```python
  from datasets import load_dataset

  # Load dataset
  dataset = load_dataset('Yuanshi/Subjects200K')
  ```
- Sample Format:
  | Key name             | Type    | Description                                                                                                                                                                                                |
  | -------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `image`              | `image` | A composite image containing a pair of images with 16-pixel padding.                                                                                                                                       |
  | `collection`         | `str`   | The name or identifier of the source collection.                                                                                                                                                           |
  | `quality_assessment` | `dict`  | Quality scores evaluated by the ChatGPT-4o model. Each quality dimension is rated on a scale of 0-5. Recommended for filtering out low-quality image pairs which do not have all dimensions rated above 5. |
  | `description`        | `dict`  | Detailed textual description of the image pair contents.                                                                                                                                                   |



### From AWS S3 (will be deprecated)
- Usage
  ```python
    from src.dataset import Subjects200K

    # Initialize dataset
    dataset = Subjects200K()

    # Access samples
    sample = dataset[0]
  ```

- Example code: `dataset_example.ipynb`

- Sample Format:
  - `instance`: Brief description of the subject
  - `image1`: Left image (512x512)
  - `image2`: Right image (512x512)
  - `description1`: Text description for left image
  - `description2`: Text description for right image
  - `image_pair`: Combined image (1024x512)


## Contributing
We welcome contributions! Please feel free to submit a Pull Request or open an Issue.

## Citation
```
@article{
  tan2024omini,
  title={OminiControl: Minimal and Universal Control for Diffusion Transformer},
  author={Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang},
  journal={arXiv preprint arXiv:2411.15098},
  year={2024}
}
```
