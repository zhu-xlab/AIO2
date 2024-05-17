# AIO2

This repository is for the codes of the TGRS paper "[AIO2: Online Correction of Object Labels for Deep Learning with Incomplete Annotation in Remote Sensing Image Segmentation](https://ieeexplore.ieee.org/document/10460569)"

## Memorization effects
The concept of *Memorization effects* was originally proposed to imply a two-stage training with noisy labels. In the first early-learning stage, model performance is continuously improved by dominant learning from most of the accurately labeled samples. In the later memorization stage, model performance begins to be degraded for overfitting to label noise information.

In this work, we reinterpret this phenomenon as a **three-stage training** with noisy labels, adding a **transition stage** between the early-learning and memorization stages, where the model performance plateaus before overfitting to label noise, as shown in Fig. 1. In the transition stage, the model reaches the highest potential when directly trained with noisy labels. Correspondingly, <ins>the training accuracy (wrt noisy labels) increases much faster both before and after the transition stage, which provides us the potential to monitor the growth rate of the training accuracy curve for detection.</ins>


![Illustration of AIO2](media/Fig-1-memorization-effects.jpg)

## Methodology
Based on the above observation, we proposed AIO2 for object-level incomplete label sets, which is mainly composed of two modules, the **Adaptive Correction Trigger (ACT)** module and the **Online Object-wise label Correction (O2C)** module:
- **ACT** is used to adaptively ceases the training when the model starts overfitting to noisy labels in the warm-up (detecting the transition stage during the training with noisy labels). After that, O2C comes into force for sample correction.
- **O2C** selects pseudo label candidates in an object-wise fashion, using a smooth filter to generate soft boundaries for candidate pseudo objects. This is the major improvement of O2C compared to commonly used pixel-wise correction strategies. Additionally, ``online'' in the O2C name includes one-off label correction at each iteration without saving historical correction results. This is another major difference from commonly used pixel-wise correction strategies, which correct labels incrementally.

In addition, a teacher model is introduced whose weights are updated by *exponential moving average (EMA)* on historical weights of the student model. The teacher model on the one hand provides more smooth training accuracy curves for the ACT module to automatically terminate the warm-up phase and simultaneously trigger O2C for label cleansing, and on the other hand serves as the pseudo label source in O2C, and thus is able to partly decouple the label correction process from model training.

![Illustration of AIO2](media/Flowchart.png)

## Citation:
If you find the codes are useful, please cite our work:
```BibTeX
@ARTICLE{liu-aio2,
  author={Liu, Chenying and Albrecht, Conrad M and Wang, Yi and Li, Qingyu and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={AIO2: Online Correction of Object Labels for Deep Learning with Incomplete Annotation in Remote Sensing Image Segmentation}, 
  year={2024},
  volume={},
  number={},
  pages={in process},
  doi={10.1109/TGRS.2024.3373908}}
```
## Usage
Scripts denoted with `png` are for the Massachusetts dataset, while those with `h5` are for the Germany dataset.
- **Label noise injection**: please refer to `data_preparation/ReadMe.txt` for details
- **Run AIO2**: an example on the Massachusetts dataset
```bash
python py_scripts/train_unet_png_emaCorrect_singleGPU.py \
        --data_path path-to-data \
        --noise_dir_name ns-dir-name \
        --monte_carlo_run 1 \
        --save_dir path-to-save \
        --model_type unet \
        --loss_type cd \
        --batch_size 128 \
        --test_batch_size 50 \
        --num_workers 8 \
        --epochs 325 \
        --learning_rate 0.001 \
        --cal_tr_acc \
        --project_name wandb-project-name \
        --entity_name wandb-entity-name \
        --wandb_mode online \
        --el_window_sizes 10 20 30 40 \
        --correct_base iter \
        --correct_model teacher \
        --correct_meth object \
        --soft_filter_size 5 \
        --display_interval 20 \
        --save_interval 5 \
        --seed 42 
```
- **Other compared methods**:
  - Baseline: `train_unet_{png/h5_smp}_pixelCorrect_singleGPU.py`
  - Pixel-wise `correction: train_unet_{png/h5_smp}_pixelCorrect_singleGPU`
  - Bootstrapping: `train_unet_{png/h5_smp}_bootstrap_singleGPU.py`
  - Consistency constraint: `train_unet_{png/h5_smp}_emaConsistReg_singleGPU.py`