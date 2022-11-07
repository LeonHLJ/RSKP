# RSKP

> **Weakly Supervised Temporal Action Localization via Representative Snippet Knowledge Propagation (CVPR 2022)**<br> 
> Linjiang Huang (CUHK), Liang Wang (CASIA), Hongsheng Li (CUHK)
>
> [![arXiv](https://img.shields.io/badge/arXiv-2203.02925-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2203.02925) [![CVPR2022](https://img.shields.io/badge/CVPR-2022-brightgreen.svg?style=plastic)](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Weakly_Supervised_Temporal_Action_Localization_via_Representative_Snippet_Knowledge_Propagation_CVPR_2022_paper.pdf) 

## Overview
The experimental results on THUMOS14 are as below.

| Method \ mAP(%) | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 | AVG |
|:----------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| [UntrimmedNet](https://arxiv.org/abs/1703.03329) | 44.4 | 37.7 | 28.2 | 21.1 | 13.7 | - | - | - |
| [STPN](https://arxiv.org/abs/1712.05080) | 52.0 | 44.7 | 35.5 | 25.8 | 16.9 | 9.9 | 4.3 | 27.0 |
| [W-TALC](https://arxiv.org/abs/1807.10418) | 55.2 | 49.6 | 40.1 | 31.1 | 22.8 | - | 7.6 | - |
| [AutoLoc](https://arxiv.org/abs/1807.08333) | - | - | 35.8 | 29.0 | 21.2 | 13.4 | 5.8 | - | - | - |
| [CleanNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Weakly_Supervised_Temporal_Action_Localization_Through_Contrast_Based_Evaluation_Networks_ICCV_2019_paper.html) | - | - | 37.0 | 30.9 | 23.9 | 13.9 | 7.1 | - |
| [MAAN](https://arxiv.org/abs/1905.08586) | 59.8 | 50.8 | 41.1 | 30.6 | 20.3 | 12.0 | 6.9 | 31.6 |
| [CMCS](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Completeness_Modeling_and_Context_Separation_for_Weakly_Supervised_Temporal_Action_CVPR_2019_paper.pdf) | 57.4 | 50.8 | 41.2 | 32.1 | 23.1 | 15.0 | 7.0 | 32.4 |
| [BM](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_Weakly-Supervised_Action_Localization_With_Background_Modeling_ICCV_2019_paper.pdf) | 60.4 | 56.0 | 46.6 | 37.5 | 26.8 | 17.6 | 9.0 | 36.3 |
| [RPN](https://ojs.aaai.org/index.php/AAAI/article/view/6760/6614) | 62.3 | 57.0 | 48.2 | 37.2 | 27.9 | 16.7 | 8.1 | 36.8 |
| [DGAM](https://dl.acm.org/doi/pdf/10.1145/3343031.3351044) | 60.0 | 54.2 | 46.8 | 38.2 | 28.8 | 19.8 | 11.4 | 37.0 |
| [TSCN](https://arxiv.org/pdf/2010.11594) | 63.4 | 57.6 | 47.8 | 37.7 | 28.7 | 19.4 | 10.2 | 37.8 |
| [EM-MIL](https://arxiv.org/abs/1911.09963) | 59.1 | 52.7 | 45.5 | 36.8 | 30.5 | 22.7 | **16.4** | 37.7 |
| [BaS-Net](https://arxiv.org/abs/1911.09963) | 58.2 | 52.3 | 44.6 | 36.0 | 27.0 | 18.6 | 10.4 | 35.3 |
| [A2CL-PT](https://arxiv.org/pdf/2007.06643) | 61.2 | 56.1 | 48.1 | 39.0 | 30.1 | 19.2 | 10.6 | 37.8 |
| [ACM-BANet](https://dl.acm.org/doi/pdf/10.1145/3394171.3413687) | 64.6 | 57.7 | 48.9 | 40.9 | 32.3 | 21.9 | 13.5 | 39.9 |
| [HAM-Net](https://arxiv.org/pdf/2101.00545) | 65.4 | 59.0 | 50.3 | 41.1 | 31.0 | 20.7 | 11.1 | 39.8 |
| [ACSNet](https://arxiv.org/pdf/2103.15088) | - | - | 51.4 | 42.7 | 32.4 | 22.0 | 11.7 | - |
| [WUM](https://arxiv.org/abs/2006.07006) | 67.5 | 61.2 | 52.3 | 43.4 | 33.7 | 22.9 | 12.1 | 41.9 |
| [AUMN](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Action_Unit_Memory_Network_for_Weakly_Supervised_Temporal_Action_Localization_CVPR_2021_paper.pdf) | 66.2 | 61.9 | 54.9 | 44.4 | 33.3 | 20.5 | 9.0 | 41.5 |
| [CoLA](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_CoLA_Weakly-Supervised_Temporal_Action_Localization_With_Snippet_Contrastive_Learning_CVPR_2021_paper.pdf) | 66.2 | 59.5 | 51.5 | 41.9 | 32.2 | 22.0 | 13.1 | 40.9 |
| [ASL](https://arxiv.org/abs/2006.07006) | 67.0 | - | 51.8 | - | 31.1 | - | 11.4 | - |
| [TS-PCA](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_The_Blessings_of_Unlabeled_Background_in_Untrimmed_Videos_CVPR_2021_paper.pdf) | 67.6 | 61.1 | 53.4 | 43.4 | 34.3 | 24.7 | 13.7 | 42.6 |
| [UGCT](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Uncertainty_Guided_Collaborative_Training_for_Weakly_Supervised_Temporal_Action_Detection_CVPR_2021_paper.pdf) | 69.2| 62.9 | 55.5 | 46.5 | 35.9 | 23.8 | 11.4 | 43.6 |
| [CO2-Net](https://dl.acm.org/doi/pdf/10.1145/3474085.3475298?casa_token=JfCwbqapIZkAAAAA:H5UGwBVZLjNB4D4Ed7eDAj2RJAq6qPETCo494_cestuwSRbADOq7SpP3-AbF3XTG2cphvsCWiF2u) | 70.1 | 63.6 | 54.5 | 45.7 | **38.3** | **26.4** | 13.4 | 44.6 |
| [D2-Net](https://openaccess.thecvf.com/content/ICCV2021/papers/Narayan_D2-Net_Weakly-Supervised_Action_Localization_via_Discriminative_Embeddings_and_Denoised_Activations_ICCV_2021_paper.pdf) | 65.7 | 60.2 | 52.3 | 43.4 | 36.0 | - | - | - |
| [FAC-Net](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Foreground-Action_Consistency_Network_for_Weakly_Supervised_Temporal_Action_Localization_ICCV_2021_paper.pdf) | 67.6 | 62.1 | 52.6 | 44.3 | 33.4 | 22.5 | 12.7 | 42.2 |
| **Ours** | **71.3** | **65.3** | **55.8** | **47.6** | 38.2 | 25.4 | 12.5 | **45.1** |

## Prerequisites
### Recommended Environment
* Python 3.6
* Pytorch 1.5
* Tensorboard Logger
* CUDA 10.1

**Note**: Our code works with different PyTorch and CUDA versions, for high version of Pytorch, you need to change one line of our code according to this [issue](https://github.com/LeonHLJ/RSKP/issues/2).

### Data Preparation
1. Prepare [THUMOS'14](https://www.crcv.ucf.edu/THUMOS14/) dataset.
    - We recommend using features and annotations provided by [this repo](https://github.com/sujoyp/wtalc-pytorch).

2. Place the [features](https://emailucr-my.sharepoint.com/personal/sujoy_paul_email_ucr_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsujoy%5Fpaul%5Femail%5Fucr%5Fedu%2FDocuments%2Fwtalc%2Dfeatures&ga=1) and [annotations](https://github.com/sujoyp/wtalc-pytorch/tree/master/Thumos14reduced-Annotations) inside a `dataset/Thumos14reduced/` folder.

## Usage

### Training
You can easily train the model by running the provided script.

- Refer to `options.py`. Modify the argument of `dataset-root` to the path of your `dataset` folder.

- Run the command below.

~~~~
$ python main.py --run-type 0 --model-id 1
~~~~

Models are saved in `./ckpt/dataset_name/model_id/`

### Evaulation

#### 
The trained model can be found [here](https://drive.google.com/file/d/1xgyebFW75B08hJrXarmK5ZQ6F5iahg4A/view?usp=sharing). (This saved model's result is slightly different from the one reported in our paper.)

Please put it into `./ckpt/dataset_name/model_id/`.

- Run the command below.

~~~~
$ python main.py --pretrained --run-type 1 --model-id 1 --load-epoch xxx
~~~~

Please refer to the log in the same folder of saved models to set the load epoch of the best model.
Make sure you set the right `model-id` that corresponds to the `model-id` during training.

## References
We referenced the repos below for the code.

* [STPN](https://github.com/bellos1203/STPN)
* [W-TALC](https://github.com/sujoyp/wtalc-pytorch)
* [BaS-Net](https://github.com/Pilhyeon/BaSNet-pytorch)

## Citation
~~~~
@InProceedings{rskp,
  title={Weakly Supervised Temporal Action Localization via Representative Snippet Knowledge Propagation},
  author={Huang, Linjiang and Wang, Liang and Li, Hongsheng},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
~~~~

## Contact
If you have any question or comment, please contact the first author of the paper - Linjiang Huang (ljhuang524@gmail.com).
