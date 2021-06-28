#!/usr/bin/env bash
set -x
nvidia-smi

python3 -c "import torch; print(torch.__version__)"

src=vi
tgt=en
bedropout=0.5
ARCH=transformer_wmt_en_de
DATAPATH=iwslt_vi_en
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}
mkdir -p $SAVEDIR
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler"
else
warmup=""
fi
warmup=""
python train.py $DATAPATH \
--tpu -a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
--dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 150000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --share-all-embeddings $warmup \
--encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout \
--bert-model-name vinai/phobert-base | tee -a $SAVEDIR/training.log
