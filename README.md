# Music Source Restoration

This guide is intended to explain how to run the project to achieve the competition results.

Special thanks to the Music-Source-Separation-Training GitHub project organized by ZFTurbo for providing pretrained model weights and relevant training and inference code. Our code is modified based on the code in that repository. The project is available at: [https://github.com/ZFTurbo/Music-Source-Separation-Training/tree/main](https://github.com/ZFTurbo/Music-Source-Separation-Training/tree/main). You can visit the link to find more model weights and other content related to MSS.

> **Note:** The following steps may have path compatibility issues. If you encounter any issues, please check the file paths in the code to ensure they are correct for execution.

## 1. Install Environment

We strongly recommend using Python 3.10 and setting up two separate environments for the two running libraries (You can use conda to install the environments).

### 1.1 Initialize MSRKit Environment

First, activate the conda environment configured for MSRKit:

```bash
cd MSRKit
pip install -r requirements.txt
```

### 1.2 Initialize AoMSS Environment

Next, activate the conda environment configured for AoMSS:

```bash
cd ../AoMSS
pip install -r requirements.txt
```

## 2. Prepare Data and Pretrained Weights

### 2.1 Download Data

Please download the competition test dataset or your own data and create a `data` folder in the root directory to place the data inside. (The code is adjusted to output results in the default format for the competition test dataset. If you're using other data, you will need to modify the data path and related logic in the code.)

### 2.2 Download Pretrained Weights

Please create a `pretrain` folder in the root directory and place the downloaded weights in the `pretrain` folder using the following naming conventions:

- **BSRNN (baseline) weights**  
  Download link: [BSRNN Weights](https://huggingface.co/yongyizang/MSRChallengeBaseline)  
  After downloading, place them in `pretrain/baseline` (including the `config` file).

- **BSRoformer weights and config files**  
  Download links:  
  [config_bs_roformer_384_8_2_485100.yaml](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml)  
  [model_bs_roformer_ep_17_sdr_9.6568.ckpt](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt)  
  After downloading, place them in `pretrain/BSRoformer`.

- **Drumsep (mdx23c) weights and config files**  
  Download links:  
  [config_mdx23c.yaml](https://github.com/jarredou/models/releases/download/DrumSep/config_mdx23c.yaml)  
  [drumsep_5stems_mdx23c_jarredou.ckpt](https://github.com/jarredou/models/releases/download/DrumSep/drumsep_5stems_mdx23c_jarredou.ckpt)  
  After downloading, place them in `pretrain/Drumsep`.

## 3. Output Final Results

### 3.1 Run BSRoformer for Initial Separation

First, run BSRoformer to perform the initial separation of the input data. In your AoMSS environment, run:

```bash
./BSRoformer_infer.sh
```

### 3.2 Run Baseline for Final Results

Then, go to the MSRKit directory, activate the MSRKit environment, and run:

```bash
./Baseline_infer.sh
```

This will produce the final results.
