# LersGAN
LersGAN: A GAN-Based Model for Low-Light Remote Sensing Image Enhancement

## Representitive Results
![representive_results](/assets/show_0.png)

## demo
![representive_results](/assets/show_1.png)

## :postbox: News

📢 **2025-4-15** --- 👋👋👋 Release codes and models for LersGAN.

📢 **2025-4-14** --- Initialized the git project.  

### TODO
- [ ] Update training code.
- [ ] Update testing code.
- [ ] Update baseline methods.
- [ ] Update visualization code for linux.

## :round_pushpin: Get You Ready

<details>
  ### :rocket:Environment Preparing
  ```
  python3.5
  ```
  You should prepare at least 3 1080ti gpus or change the batch size. 
  
  ```pip install -r requirement.txt``` </br>
  ```mkdir model``` </br>
  Download VGG pretrained model from [[Google Drive 1]](https://drive.google.com/file/d/1IfCeihmPqGWJ0KHmH-mTMi_pn3z3Zo-P/view?usp=sharing), and then put it into the directory `model`.
  
</details>

## :dancers: Training process
<details>
  TODO
<!--   Before starting training process, you should launch the `visdom.server` for visualizing.
  
  ```nohup python -m visdom.server -port=8097```
  
  then run the following command
  
  ```python scripts/script.py --train``` -->
</details>

## :space_invader: Testing process
<details>  
  TODO
<!--   Create directories `../test_dataset/testA` and `../test_dataset/testB`. Put your test images on `../test_dataset/testA` (And you should keep whatever one image in `../test_dataset/testB` to make sure program can start.)
  Download [pretrained model](https://drive.google.com/file/d/1AkV-n2MdyfuZTFvcon8Z4leyVb0i7x63/view?usp=sharing) and put it into `./checkpoints/enlightening`
  Run
  ```python scripts/script.py --predict ``` -->
</details>
