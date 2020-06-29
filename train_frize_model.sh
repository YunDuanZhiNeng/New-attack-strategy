# train frize model after softermax training
#python3  softmax_training_frize.py  --max-epoch=100 --t-max=100  --gpu=0  --lr=0.002 --eval-freq=10 --save-filename=Only_Softmax_1024_avg_pool --filename=Models_Softmax/CIFAR10_Softmax.pth.tar

# train frize model after pcl training
#python3  softmax_training_frize.py  --max-epoch=100 --t-max=100  --gpu=2  --lr=0.002 --eval-freq=10 --save-filename=Only_PCL_1024_avg_pool --filename=Models_PCL/CIFAR10_PCL.pth.tar

# train frize model after pcl and pgd training
python3  softmax_training_frize.py  --max-epoch=100 --t-max=100  --gpu=0 --eval-freq=10 --lr=0.002 --save-filename=robust_model_original_1024_adam_0.0_distill_full_  --filename=robust_model.pth.tar
