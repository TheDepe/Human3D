#!/bin/bash
conda activate human3d_cuda113
# For CUDA
export LD_LIBRARY_PATH=/is/software/nvidia/cuda-11.4/lib64
export PATH=$PATH:/is/software/nvidia/cuda-11.4/bin
export CUDA_HOME=/is/software/nvidia/cuda-11.4

# For cuDNN
export C_INCLUDE_PATH=/is/software/nvidia/cudnn-8.2-cu11.4/include
export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH
export LIBRARY_PATH=/is/software/nvidia/cudnn-8.2-cu11.4/lib64
export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH

python main.py \
general.experiment_name="Mask3D_horse_big_run_eval" \
general.project_name="mask3d_horse_seg" \
data/datasets=synthetic_horses \
general.num_targets=3 \
data.num_labels=2 \
model=mask3d \
loss=set_criterion \
model.num_queries=1 \
trainer.check_val_every_n_epoch=1 \
general.topk_per_image=-1 \
model.non_parametric_queries=false \
trainer.max_epochs=36 \
data.batch_size=4 \
data.num_workers=10 \
general.reps_per_epoch=1 \
model.config.backbone._target_=models.Res16UNet18B \
data.part2human=true \
loss.num_classes=2 \
model.num_classes=2 \
callbacks=callbacks_instance_segmentation_horse \
general.checkpoint="/ssd-disk/data_ssd/VAREN/models/Mask3d/fine_turned_horse_model.ckpt" \
general.train_mode=false \
general.save_visualizations=true \
+general.evaluate=false # set to false in order to just run test
