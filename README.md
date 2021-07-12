# AutoEncoder implementation for SMI21 Project Area (Telluride)

## Prerequisites
- Python 3.8
- PyTorch
- tqdm
- torchvision
- torchsummary
- ...

## Installation
```bash
git clone https://github.com/Barchid/SMI21-AutoEncoder
cd SMI21-AutoEncoder
pip install torch torchvision tqdm torchsummary tensorboard
mkdir datasets
```

Then, download Michael Furlong's dataset in the SMI shared folder (Google Drive), unzip the archive file in the `datasets/` folder and remove the `event_imgs` subfolder (`rm -rf datasets/training_data/event_imgs`).

## Debug model
Open `debug.py` script and change the model you want to debug at the beginning of the file. Then you can run:

```bash
python debug.py
```

It will try to overfit a single batch in order to check if the loss drops to 0. If it does not, it means the model is likely to have some bugs in it.


## Train
```bash
python train.py \
--experiment='MyAwesomeExperiment' \
--data=datasets/training_data \
--arch=cornets \
--epochs=200 \
--batch-size=16 \
--lr=1e-3 \
--print-freq=10 \
--gpu=0
```

If you want to know every options available, type `python train.py -h`.

