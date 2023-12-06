CUDA_VISIBLE_DEVICES=0 \
python3 \
train.py \
--batchsize 8 \
--savepath "../model" \
--datapath "../data/TrainDataset" \
--lr 0.03 \
--epoch 100 \
--wd 5e-4 \
--fr 0.006