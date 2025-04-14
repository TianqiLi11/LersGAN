# LersGAN
LersGAN: A GAN-Based Model for Low-Light Remote Sensing Image Enhancement

### Representitive Results
![representive_results](/assets/show_3.png)

### Overal Architecture
![architecture](/assets/arch.png)

## Environment Preparing
```
python3.5
```
You should prepare at least 3 1080ti gpus or change the batch size. 


```pip install -r requirement.txt``` </br>
```mkdir model``` </br>
Download VGG pretrained model from [[Google Drive 1]](https://drive.google.com/file/d/1IfCeihmPqGWJ0KHmH-mTMi_pn3z3Zo-P/view?usp=sharing), and then put it into the directory `model`.

### Training process
Before starting training process, you should launch the `visdom.server` for visualizing.

```nohup python -m visdom.server -port=8097```

then run the following command

```python scripts/script.py --train```

### Testing process

Download [pretrained model](https://drive.google.com/file/d/1AkV-n2MdyfuZTFvcon8Z4leyVb0i7x63/view?usp=sharing) and put it into `./checkpoints/enlightening`

Create directories `../test_dataset/testA` and `../test_dataset/testB`. Put your test images on `../test_dataset/testA` (And you should keep whatever one image in `../test_dataset/testB` to make sure program can start.)

Run

```python scripts/script.py --predict ```






# MoMask: Generative Masked Modeling of 3D Human Motions (CVPR 2024)
### [[Project Page]](https://ericguo5513.github.io/momask) [[Paper]](https://arxiv.org/abs/2312.00063) [[Huggingface Demo]](https://huggingface.co/spaces/MeYourHint/MoMask) [[Colab Demo]](https://github.com/camenduru/MoMask-colab)
![teaser_image](https://ericguo5513.github.io/momask/static/images/teaser.png)

If you find our code or paper helpful, please consider starring our repository and citing:
```
@inproceedings{guo2024momask,
  title={Momask: Generative masked modeling of 3d human motions},
  author={Guo, Chuan and Mu, Yuxuan and Javed, Muhammad Gohar and Wang, Sen and Cheng, Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1900--1910},
  year={2024}
}
```

## :postbox: News
📢 **2024-08-02** --- The [WebUI demo 🤗](https://huggingface.co/spaces/MeYourHint/MoMask) is now running smoothly on a CPU. No GPU is required to use MoMask.

📢 **2024-02-26** --- 🔥🔥🔥 Congrats! MoMask is accepted to CVPR 2024.

📢 **2024-01-12** --- Now you can use MoMask in Blender as an add-on. Thanks to [@makeinufilm](https://twitter.com/makeinufilm) for sharing the [tutorial](https://medium.com/@makeinufilm/notes-on-how-to-set-up-the-momask-environment-and-how-to-use-blenderaddon-6563f1abdbfa).

📢 **2023-12-30** --- For easy WebUI BVH visulization, you could try this website [bvh2vrma](https://vrm-c.github.io/bvh2vrma/) from this [github](https://github.com/vrm-c/bvh2vrma?tab=readme-ov-file).

📢 **2023-12-29** --- Thanks to Camenduru for supporting the [🤗Colab](https://github.com/camenduru/MoMask-colab) demo.

📢 **2023-12-27** --- Release WebUI demo. Try now on [🤗HuggingFace](https://huggingface.co/spaces/MeYourHint/MoMask)!

📢 **2023-12-19** --- Release scripts for temporal inpainting.

📢 **2023-12-15** --- Release codes and models for momask. Including training/eval/generation scripts.

📢 **2023-11-29** --- Initialized the webpage and git project.  


## :round_pushpin: Get You Ready

<details>
  
### 1. Conda Environment
```
conda env create -f environment.yml
conda activate momask
pip install git+https://github.com/openai/CLIP.git
```
We test our code on Python 3.7.13 and PyTorch 1.7.1

#### Alternative: Pip Installation
<details>
We provide an alternative pip installation in case you encounter difficulties setting up the conda environment.

```
pip install -r requirements.txt
```
We test this installation on Python 3.10

</details>

### 2. Models and Dependencies

#### Download Pre-trained Models
```
bash prepare/download_models.sh
```

#### Download Evaluation Models and Gloves
For evaluation only.
```
bash prepare/download_evaluator.sh
bash prepare/download_glove.sh
```

#### Troubleshooting
To address the download error related to gdown: "Cannot retrieve the public link of the file. You may need to change the permission to 'Anyone with the link', or have had many accesses". A potential solution is to run `pip install --upgrade --no-cache-dir gdown`, as suggested on https://github.com/wkentaro/gdown/issues/43. This should help resolve the issue.

#### (Optional) Download Manually
Visit [[Google Drive]](https://drive.google.com/drive/folders/1sHajltuE2xgHh91H9pFpMAYAkHaX9o57?usp=drive_link) to download the models and evaluators mannually.

### 3. Get Data

You have two options here:
* **Skip getting data**, if you just want to generate motions using *own* descriptions.
* **Get full data**, if you want to *re-train* and *evaluate* the model.

**(a). Full data (text + motion)**

**HumanML3D** - Follow the instruction in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then copy the result dataset to our repository:
```
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```
**KIT**-Download from [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then place result in `./dataset/KIT-ML`

#### 

</details>

## :rocket: Demo
<details>

### (a) Generate from a single prompt
```
python gen_t2m.py --gpu_id 1 --ext exp1 --text_prompt "A person is running on a treadmill."
```
### (b) Generate from a prompt file
An example of prompt file is given in `./assets/text_prompt.txt`. Please follow the format of `<text description>#<motion length>` at each line. Motion length indicates the number of poses, which must be integeter and will be rounded by 4. In our work, motion is in 20 fps.

If you write `<text description>#NA`, our model will determine a length. Note once there is **one** NA, all the others will be **NA** automatically.

```
python gen_t2m.py --gpu_id 1 --ext exp2 --text_path ./assets/text_prompt.txt
```


A few more parameters you may be interested:
* `--repeat_times`: number of replications for generation, default `1`.
* `--motion_length`: specify the number of poses for generation, only applicable in (a).

The output files are stored under folder `./generation/<ext>/`. They are
* `numpy files`: generated motions with shape of (nframe, 22, 3), under subfolder `./joints`.
* `video files`: stick figure animation in mp4 format, under subfolder `./animation`.
* `bvh files`: bvh files of the generated motion, under subfolder `./animation`.

We also apply naive foot ik to the generated motions, see files with suffix `_ik`. It sometimes works well, but sometimes will fail.
  
</details>

## :dancers: Visualization
<details>

All the animations are manually rendered in blender. We use the characters from [mixamo](https://www.mixamo.com/#/). You need to download the characters in T-Pose with skeleton.

### Retargeting
For retargeting, we found rokoko usually leads to large error on foot. On the other hand, [keemap.rig.transfer](https://github.com/nkeeline/Keemap-Blender-Rig-ReTargeting-Addon/releases) shows more precise retargetting. You could watch the [tutorial](https://www.youtube.com/watch?v=EG-VCMkVpxg) here.

Following these steps:
* Download keemap.rig.transfer from the github, and install it in blender.
* Import both the motion files (.bvh) and character files (.fbx) in blender.
* `Shift + Select` the both source and target skeleton. (Do not need to be Rest Position)
* Switch to `Pose Mode`, then unfold the `KeeMapRig` tool at the top-right corner of the view window.
* For `bone mapping file`, direct to `./assets/mapping.json`(or `mapping6.json` if it doesn't work), and click `Read In Bone Mapping File`. This file is manually made by us. It works for most characters in mixamo.
* (Optional) You could manually fill in the bone mapping and adjust the rotations by your own, for your own character. `Save Bone Mapping File` can save the mapping configuration in local file, as specified by the mapping file path.
* Adjust the `Number of Samples`, `Source Rig`, `Destination Rig Name`.
* Clik `Transfer Animation from Source Destination`, wait a few seconds.

We didn't tried other retargetting tools. Welcome to comment if you find others are more useful.

### Scene

We use this [scene](https://drive.google.com/file/d/16SbrnG9JsJ2w7UwCFmh10PcBdl6HxlrA/view?usp=drive_link) for animation.


</details>

## :clapper: Temporal Inpainting
<details>
We conduct mask-based editing in the m-transformer stage, followed by the regeneration of residual tokens for the entire sequence. To load your own motion, provide the path through `--source_motion`. Utilize `-msec` to specify the mask section, supporting either ratio or frame index. For instance, `-msec 0.3,0.6` with `max_motion_length=196` is equivalent to `-msec 59,118`, indicating the editing of the frame section [59, 118]. 

```
python edit_t2m.py --gpu_id 1 --ext exp3 --use_res_model -msec 0.4,0.7 --text_prompt "A man picks something from the ground using his right hand."
```

Note: Presently, the source motion must adhere to the format of a HumanML3D dim-263 feature vector. An example motion vector data from the HumanML3D test set is available in `example_data/000612.npy`. To process your own motion data, you can utilize the `process_file` function from `utils/motion_process.py`.

</details>

## :space_invader: Train Your Own Models
<details>
**Note**: You have to train RVQ **BEFORE** training masked/residual transformers. The latter two can be trained simultaneously.
</details>

## :book: Evaluation
<details>
