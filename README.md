SynFlow: Scaling Up LiDAR Scene Flow Estimation with Synthetic Data
---

[![arXiv](https://img.shields.io/badge/arXiv-2604.09411-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.09411)
[![page](https://img.shields.io/badge/Project-Page-green)](https://kin-zhang.github.io/SynFlow)
<!-- [![pdfreview](https://img.shields.io/badge/OpenReview-PDF-blue)](https://github.com/Kin-Zhang/SynFlow/discussions/1) -->
<!-- [![poster](https://img.shields.io/badge/CVPR'26|Poster-6495ed?style=flat&logo=Shotcut&logoColor=wihte)](https://drive.google.com/file/d/1RNwMUiw1lEZ9DRPAZrBb6f0geLEJUCe2/view?usp=sharing) -->
<!-- [![video](https://img.shields.io/badge/Presentation-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/BV47IUSEOgE) -->

<p align="center">
  <img alt="synflow_cover" src="https://kin-zhang.github.io/SynFlow/assets/images/cover.png" />
</p>

SynFlow got accepeted in ECCV2026, I'm updating the repo and README, stay tuned for the dataset release and code release! Timeline and TODO:
- [x] 2026-06-18: Initial the repo and add README.
- [ ] Update the CARLA code for dataset generation and add the dataset generation instruction in README.
- [ ] Upload the dataset to Huggingface and add the download link in README.
- [ ] Add review comment and rebuttal pdf and poster link


## Prerequisites

### CARLA Installation


## Synthetic Dataset Generation

SynFlow-4k Dataset Donwload Link
| Dataset/Model |Download Link | Description |
|:---------------------:|:-------------:|:-----------------:|
| SynFlow-4k | [huggingface TODO](TODO) | It contains around 4k scenes includes 940k frames with 3D flow ground truth... TODO |
| DeltaFlow weight (trained on SynFlow-4k) | [huggingface TODO](TODO) | Model trained on SynFlow-4k dataset, which can be used for evaluation and as a pretrained model for real-world data finetuning. |
| DeltaFlow weight (trained on SynFlow-4k with real-world data) | [huggingface TODO](TODO) | Model trained on SynFlow-4k dataset with real-world data, which can be used for evaluation and as a pretrained model for real-world data finetuning. |

## Model Training and Inference

Please refer to the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow) for detail training instructions.

Here is the overall pipeline of our method:
<p align="center">
  <img alt="teflow_pipeline" src="https://github.com/user-attachments/assets/0afd5c20-2bcc-4a02-aad0-8d8daf04100f" />
</p>

1. Follow the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow/tree/main?tab=readme-ov-file#0-installation) to setup the environment or [use docker](https://github.com/KTH-RPL/OpenSceneFlow?tab=readme-ov-file#docker-recommended-for-isolation).
2. Download the SynFlow dataset and prepare the [TODO]
3. Run the training with the following command (modify the data path accordingly):
```bash
TODO
```

### Evaluation

Trained your own model or downloaded the pretrained weights from [Table](#link to above table)

Please check the local evaluation result (raw terminal output screenshot) in [this discussion thread TODO](https://github.com/Kin-Zhang/SynFlow/discussions/2). 
You can also run the evaluation by yourself with the following command with trained weights:
```bash
python eval.py checkpoint=${path_to_pretrained_weights} dataset_path=${demo_data_path}
```



### Visualization

<img width="1627" height="821" alt="image" src="https://github.com/user-attachments/assets/32957bcb-fec8-46be-a08e-c637572dde8a" />

To make your own visualizations, please refer to the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow/tree/main?tab=readme-ov-file#4-visualization) for visualization instructions.

## Cite & Acknowledgements

```
@article{zhang2026synflow,
  author    = {Zhang, Qingwen and Zhu, Xiaomeng and Jiang, Chenhan and Jensfelt, Patric},
  title     = {{SynFlow}: Scaling Up LiDAR Scene Flow Estimation with Synthetic Data},
  journal   = {arXiv preprint arXiv:2604.09411},
  year      = {2026},
}
```
This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation and Prosense (2020-02963) funded by Vinnova. 
