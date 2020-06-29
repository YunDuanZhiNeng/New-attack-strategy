export CUDA_VISIBLE_DEVICES=0
for outputs_name in  outputs256_17 outputs256_16 outputs256_15 ; do         
  for model in Models_Softmax/frize_pcl_adversarial_free_distill_full_248_yyr_new__99_89.40999603271484.pth.tar ; do
    for attack in fgsm pgd mim bim; do #fgsm;do
      for epsilon in 0.03 ; do # 0.02 0.03;do
        for scale in 1  ; do
           param="--epsilon=$epsilon --attack=$attack  --scale=$scale --file-name=$model --outputs-name=$outputs_name"
           python3 test_frize_robust.py  $param
				done
      done
    done
  done
done

