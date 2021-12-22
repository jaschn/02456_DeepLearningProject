module load python3/3.9.5
python3 -m venv instance-venv
source instance-venv/bin/activate
python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
python3 -m pip install -r requirements.txt
gdown --id 1ne3IfBasihK29PkaVpRSYrsOJ2eqyIVy
gdown --id 1vO9oLaPHtA_mj_hcVlOQi-_nLG22qZAw
gdown --id 1q1t873smbzRWae3tyUYNj1VG1FcFHHzI
unzip train_sample.zip
unzip coco_full.zip
unzip coco_instance.zip
rm -f train_sample.zip
rm -f coco_instance.zip
rm -f coco_full.zip

mkdir ./checkpoints/coco_mask
cp ./checkpoints/coco_full/latest_net_G.pth ./checkpoints/coco_mask/latest_net_GF.pth
cp ./checkpoints/coco_instance/latest_net_G.pth ./checkpoints/coco_mask/latest_net_G.pth
cp ./checkpoints/coco_full/latest_net_G.pth ./checkpoints/coco_mask/latest_net_GComp.pth