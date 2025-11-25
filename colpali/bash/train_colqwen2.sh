


accelerate launch --main_process_port 29501 modernvbert/src/modernvbert/contrastive_training/train.py \
    modernvbert/src/modernvbert/contrastive_training/ablation/alignment_objective/config_li/colvbert.yaml \
    > log/colvbert.log 2>&1 &




accelerate launch --main_process_port 29501 /var/scratch/jqiao/colpali/scripts/train/train_colbert.py \
    /var/scratch/jqiao/colpali/scripts/configs/modernvbert/train_colqmodernvbert.yaml > /var/scratch/jqiao/colpali/log/train_colqmodernvbert.log 2>&1 &
