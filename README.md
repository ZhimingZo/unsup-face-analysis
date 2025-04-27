# unsup-face-analysis
Implementation of "Towards Unsupervised Learning of Joint Facial Landmark Detection and Head Pose Estimation"

# Conda Environment Setup
conda env create -f environment.yml
conda activate imm2
export PYTHONPATH="$PYTHONPATH:."


#Data Download from Google Drive: 

please put all data into Data Folder

#Inference:

#Eval on MAFL 
python eval/eval_mafl.py  

#Eval on AFLW
python eval/eval_aflw.py 

#EVAL on BIWI
python eval/angles_eva_auto.py

#Training:
python train/train_3D_aware_celeba.py 



