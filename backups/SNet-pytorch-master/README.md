# S-Net

This repository is implementation of the "S-Net: A Scalable Convolutional Neural Network for JPEG Compression Artifact Reduction". <br />

## Requirements
- Python 3.7
- PyTorch 1.0.0
- Tensorflow 1.13.0
- tqdm 4.30.0
- Numpy 1.15.4
- Pillow 5.4.1

**Tensorflow** is required for quickly fetching image in training phase.

## Results

<table>
    <tr>
        <td><center>Input</center></td>
        <td><center>JPEG (Quality 10)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/monarch.bmp" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./data/monarch_jpeg_q10.png" height="300"></center>
    	</td>
    </tr>
    <tr>
        <td><center>AR-CNN</center></td>
        <td><center><b>S-Net - Metric 8</b></center></td>
    </tr>
    <tr>
        <td>
        	<center><img src="./data/monarch_ARCNN.png" height="300"></center>
        </td>
        <td>
        	<center><img src="./data/monarch_S-Net.png" height="300"></center>
        </td>
    </tr>
</table>

## Usages

### Train

When training begins, the model weights will be saved every epoch. <br />
If you want to train quickly, you should use **--use_fast_loader** option.

```bash
python main.py --num_metrics 8 \
               --structure_type "advanced" \  # classic, advanced
               --images_dir "" \
               --outputs_dir "" \
               --jpeg_quality 10 \
               --patch_size 48 \
               --batch_size 16 \
               --num_epochs 20 \
               --lr 1e-4 \
               --threads 8 \
               --seed 123 \
               --use_fast_loader              
```

### Test

Output results consist of image compressed with JPEG and image with artifacts reduced.

```bash
python example --num_metrics 8 \
               --structure_type "advanced" \  # classic, advanced
               --weights_path "" \
               --image_path "" \
               --outputs_dir "" \
               --jpeg_quality 10               
```
