# A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement

A minimum unofficial implementation of the [A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement (CRN)](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1405.pdf) using PyTorch.

## ToDo
- [x] Real-time version
- [x] Update trainer
- [x] Visualization of the spectrogram and the metrics (PESQ, STOI, SI-SDR) in the training
- [ ] More docs

## Usage

- Training:

```
python train.py -C config/train/vctk_model.json5
```
```
python train.py -C config/train/dns3_model.json5
```

- Inference:

```
python inference.py \
    -C config/inference/basic.json5 \
    -cp ./checkpoints/vctk_20251218/vctk_model/checkpoints/model_0500.pth \
    -dist ./result/remixed_XR_24k_crnn_20251218
```

Check out the README of [Wave-U-Net for SE](https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement) to learn more.

- TensorBoard:
```
tensorboard --logdir=checkpoints/vctk_20251218/vctk_model/logs
```

## PyTorch -> ONNX -> TFLite

- PyTorch → ONNX → TF:

```
conda create -n onnx python=3.10 -y
conda activate onnx

pip install torch==2.1.2
pip install onnx==1.14.1
pip install onnx-tf==1.10.0


pip install torch==2.3.0
pip install onnx==1.16.0
pip install onnx-tf==1.10.0
pip install tensorflow==2.15.0
pip install numpy
```

```
TORCH_ONNX_USE_EXPERIMENTAL_EXPORTER=0 python export_minicrn_causal128_onnx.py


onnx-tf convert \
  -i MiniCRN_Causal128.onnx \
  -o MiniCRN_Causal128_tf
```

- TF → TFLite:
```
conda create -n tf2tflite python=3.10 -y
conda activate tf2tflite


pip install \
  tensorflow==2.15.0 \
  numpy==1.26.4


python export_minicrn_causal128_tflite.py

```


## Performance

PESQ, STOI, SI-SDR on DEMAND - Voice Bank test dataset, for reference only:

| Experiment | PESQ | SI-SDR | STOI |
| --- | --- | --- | --- |
|Noisy | 1.979 | 8.511| 0.9258|
|CRN | 2.528| 17.71| 0.9325|
|CRN signal approximation  |2.606 |17.84 |0.9382|

## Dependencies

- Python==3.\*.\*
- torch==1.\*
- librosa==0.7.0
- tensorboard
- pesq
- pystoi
- matplotlib
- tqdm

## References

- [CRNN_mapping_baseline](https://github.com/YangYang/CRNN_mapping_baseline)
- [A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang1.interspeech18.pdf)
- [EHNet](https://github.com/ododoyo/EHNet)
- [Convolutional-Recurrent Neural Networks for Speech Enhancement](https://arxiv.org/abs/1805.00579)
