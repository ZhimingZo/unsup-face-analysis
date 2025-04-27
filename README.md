# unsup-face-analysis
Implementation of [Towards Unsupervised Learning of Joint Facial Landmark Detection and Head Pose Estimation] chrome-(https://www.cs.uic.edu/~tangw/files/2025_PR_ZhimingZou.pdf) by Zhiming Zou,  Dian Jia and Wei Tang. If you find our code useful in your research, please consider citing:


```
@article{zou2025towards,
  title={Towards unsupervised learning of joint facial landmark detection and head pose estimation},
  author={Zou, Zhiming and Jia, Dian and Tang, Wei},
  journal={Pattern Recognition},
  pages={111393},
  year={2025},
  publisher={Elsevier}
}
```

### Conda Environment Setup
```
  conda env create -f environment.yml 

  conda activate imm2

  export PYTHONPATH="$PYTHONPATH:."
```

### Data Download from Google Drive: 

please put all data into Data Folder

### Inference:

### Eval on MAFL 
```
  python eval/eval_mafl.py  
```
### Eval on AFLW
```
  python eval/eval_aflw.py 
```
### EVAL on BIWI
```
  python eval/angles_eva_auto.py
```
### Training:
```
python train/train_3D_aware_celeba.py

python train/train_3D_aware_aflw.py
```
### References
[1] Tomas et al. [Unsupervised learning of object landmarks through conditional image generation](https://proceedings.neurips.cc/paper_files/paper/2018/file/1f36c15d6a3d18d52e8d493bc8187cb9-Paper.pdf). NIPS 2018.
[2] Siva Karthik et al. [Self-Supervised Viewpoint Learning from Image Collections]([https://arxiv.org/pdf/1811.11742.pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mustikovela_Self-Supervised_Viewpoint_Learning_From_Image_Collections_CVPR_2020_paper.pdf)). CVPR 2020.


### Acknowledgement
This code is extended from the following repositories.
-[IMM] (https://github.com/tomasjakab/imm)
-[SSV] (https://github.com/NVlabs/SSV)
