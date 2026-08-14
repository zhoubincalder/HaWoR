<div align="center">

# HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos

[Jinglei Zhang]()<sup>1</sup> &emsp; [Jiankang Deng](https://jiankangdeng.github.io/)<sup>2</sup> &emsp; [Chao Ma](https://scholar.google.com/citations?user=syoPhv8AAAAJ&hl=en)<sup>1</sup> &emsp; [Rolandos Alexandros Potamias](https://rolpotamias.github.io)<sup>2</sup> &emsp;  

<sup>1</sup>Shanghai Jiao Tong University, China
<sup>2</sup>Imperial College London, UK <br>

<font color="blue"><strong>CVPR 2025 Highlight✨</strong></font> 

<a href='https://arxiv.org/abs/2501.02973'><img src='https://img.shields.io/badge/Arxiv-2501.02973-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
<a href='https://arxiv.org/pdf/2501.02973'><img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a> 
<a href='https://hawor-project.github.io/'><img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
<a href='https://github.com/ThunderVVV/HaWoR'><img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
<a href='https://huggingface.co/spaces/ThunderVVV/HaWoR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'></a>
</div>

This is the official implementation of **[HaWoR](https://hawor-project.github.io/)**, a hand reconstruction model in the world coordinates:

![teaser](assets/teaser.png)

## Installation
 
### Installation
```
git clone --recursive https://github.com/ThunderVVV/HaWoR.git
cd HaWoR
```

The Python environment is managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml`
pins Python 3.10 (chumpy, needed by smplx to unpickle the MANO models, calls the
`inspect.getargspec` that was removed in 3.11) and installs CUDA 12.8 torch wheels.

```bash
uv sync
```

That gives you everything needed to preprocess ground truth and train. Prefix commands
with `uv run` (e.g. `uv run python train.py ...`), or activate `.venv` directly.

Optional extras, installed on demand:

```bash
uv sync --extra demo    # detection, tracking and rendering for demo.py
uv sync --extra slam    # masked DROID-SLAM + Metric3D scale estimation
uv sync --extra hot3d   # HOT3D download/export toolkit
```

Two packages are deliberately kept out of the lockfile because they compile
extensions against the already-installed torch, which cannot be resolved from an
sdist. Neither is needed for training — install them only if you want the demo or
the SLAM stage:

```bash
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

```bash
uv pip install --no-build-isolation torch-scatter==2.1.2
```

<details>
<summary>Original conda instructions (PyTorch 1.13 / CUDA 11.7)</summary>

Note that torch 1.13+cu117 predates Blackwell (sm_120) GPUs and will fail there with
"no kernel image is available for execution on the device"; the uv setup above installs
cu128 wheels instead.

```bash
conda create --name hawor python=3.10
conda activate hawor

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# Install requirements
pip install -r requirements.txt
pip install pytorch-lightning==2.2.4 --no-deps
pip install lightning-utilities torchmetrics==1.4.0
```
</details>

### Install masked DROID-SLAM:

```
cd thirdparty/DROID-SLAM
python setup.py install
```

Download DROID-SLAM official weights [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing), put it under `./weights/external/`.

### Install Metric3D

Download Metric3D official weights [metric_depth_vit_large_800k.pth](https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view?usp=drive_link), put it under `thirdparty/Metric3D/weights`.

### Download the model weights

```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./weights/external/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml -P ./weights/hawor/
```
It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de). 
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and put the hand model to the `_DATA/data/mano/MANO_RIGHT.pkl` and `_DATA/data_left/mano_left/MANO_LEFT.pkl`. 

Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).
## Demo

For visualizaiton in world view, run with:
```bash
python demo.py --video_path ./example/video_0.mp4  --vis_mode world
```

For visualizaiton in camera view, run with:
```bash
python demo.py --video_path ./example/video_0.mp4 --vis_mode cam
```

## Training

This trains the **camera-space hand motion estimator** only. The SLAM stage uses
frozen off-the-shelf weights (DROID-SLAM + Metric3D) and the motion infiller is a
separate model.

### 1. Prepare the sequences

Export and preprocess sequences exactly as for evaluation (see *Evaluation on HOT3D*),
then split the sequence names into `train.json` / `val.json` under the export root.

### 2. Extract the ground truth

```bash
uv run python lib/datasets/hawor_preprocess_train.py --video_root datasets/hot3d_trainset_export --set_file train.json
```

This writes one `train_anno.npz` per sequence containing, for each hand and frame:
GT boxes, 2D joints with per-joint visibility, root-relative 3D joints, camera-space
axis-angle pose and betas. Repeat with `--set_file val.json`.

### 3. Train

```bash
uv run python train.py --cfg hawor/configs/hawor_train.yaml --video_root datasets/hot3d_trainset_export
```

Each sample is a contiguous 16-frame window of a single hand, which is what the
space-time and motion modules expect. Notes on the setup:

- The network is right-hand only. Left hands are mirrored into right-hand space by
  the dataset and trained as ordinary right hands; the `do_flip` path in
  `HAWOR.forward_step` is inference-only.
- Augmentation (scale, translation, colour) is sampled **once per window**, not per
  frame, so the temporal modules do not learn to undo synthetic jitter.
- In-plane rotation and horizontal flip augmentation are rejected by the dataset:
  rotation breaks the CLIFF bbox feature and the full-frame projection, and flipping
  a handed model is a no-op here.
- Losses reduce over the batch with `TRAIN.LOSS_REDUCTION`, default `mean` to
  match the released `model_config.yaml`. `sum` gives the HaMeR convention, where
  loss magnitude — and the effective learning rate — scales with `BATCH_SIZE * 16`.
- Use `bf16-mixed`. `16-mixed` inserts a gradient scaler, which breaks the manual
  `clip_grad_norm_(..., error_if_nonfinite=True)` in `training_step`; `train.py`
  refuses that combination.

By default the ViT-H backbone is loaded from a pretrained checkpoint and frozen
(and kept in `eval()` mode, since it is built with `drop_path_rate=0.55`). To
fine-tune end-to-end afterwards, set `MODEL.BACKBONE.FREEZE: False`.

### Relationship to the released recipe

`weights/hawor/model_config.yaml` ships with the pretrained weights and records how
the released model was trained. This config matches it on architecture
(`ST_HDIM`/`ST_NLAYER`/`MOTION_HDIM`/`MOTION_NLAYER`, `IMAGE_SIZE`), `LOSS_WEIGHTS`,
`LR`, `WEIGHT_DECAY`, `BATCH_SIZE` and `LOSS_REDUCTION` — the architecture match is
verified by loading `hawor.ckpt` into it with `strict=True`.

Two deliberate differences:

- **Rotation augmentation.** The release used `ROT_FACTOR: 30`, `ROT_AUG_RATE: 0.6`.
  This reimplementation leaves it off, because rotating the crop only stays exact if
  the principal point moves to the crop centre, which would zero out the CLIFF bbox
  feature that conditions the model. The dataset raises rather than silently applying
  an inconsistent transform.
- **Single dataset.** The release trained `MULTI_SET` over HOT3D, ARCTIC, DexYCB and
  HO3D at equal weight. This pipeline takes one export root; matching the release
  means exporting the other three in the same layout and concatenating them.

## Evaluation on HOT3D

### Download HOT3D

Get `Hot3DAria_download_urls.json` and `Hot3DAssets_download_urls.json` from [hot3d website](https://www.projectaria.com/datasets/hot3D/) and put them under `hot3d/data_downloader/`.

Download a copy of MANO offical website model(`mano_v1_2.zip`) and put them to `hot3d/mano_v1_2`

```
cd hot3d/data_downloader
python3 dataset_downloader_base_main.py -c Hot3DAssets_download_urls.json -o ../dataset --sequence_name all
python3 dataset_downloader_base_main.py -c Hot3DAria_download_urls.json -o ../dataset --data_types all --sequence_name P0001_a68492d5 P0001_9b6feab7 P0014_8254f925 P0011_76ea6d47 P0014_84ea2dcc P0001_8d136980 P0012_476bae57 P0012_130a66e1 P0014_24cb3bf0 P0010_1c9fe708 P0002_2ea9af5b P0011_11475e24 P0010_0ecbf39f P0010_160e551c P0015_42b8b389 P0012_915e71c6 P0002_65085bfc P0011_47878e48 P0011_cee8fe4f P0002_016222d1 P0012_d85e10f6 P0012_119de519 P0010_41c4c626 P0012_f7e3880b P0009_02511c2f P0011_72efb935 P0010_924e574e 
```

*: Downloading and processing code under `hot3d/` is adapted from [Official HOT3D Toolkit](https://github.com/facebookresearch/hot3d).

### Extract HOT3D GT
```
mkdir datasets
cd hot3d
python export_gt.py
mv hot3d_dataset_export ../datasets/hot3d_valset_export
```

### Preprocess
```
python lib/datasets/hot3d_dataset_preprocess.py --video_root datasets/hot3d_valset_export --set_file val.json --for_eval
```

### Eval

Run hand motion estimation:

```
python scripts/scripts_eval/eval_hawor_hot3d.py --inference_stage --gen_hand_mask
```

Then run SLAM stage:

```
python scripts/scripts_eval/test_mdslam_hot3d.py

```

Evaluation:

```
python scripts/scripts_eval/eval_hawor_hot3d.py --eval_stage
```

## Evaluation on DexYCB

DexYCB evaluation code (not cleaned) is available at https://github.com/ThunderVVV/dex-ycb-toolkit .


## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [HaMeR](https://github.com/geopavlakos/hamer/)
- [WiLoR](https://github.com/rolpotamias/WiLoR)
- [SLAHMR](https://github.com/vye16/slahmr)
- [TRAM](https://github.com/yufu-wang/tram)
- [CMIB](https://github.com/jihoonerd/Conditional-Motion-In-Betweening)


## License 
HaWoR models fall under the [CC-BY-NC--ND License](./license.txt). This repository depends also on [MANO Model](https://mano.is.tue.mpg.de/license.html), which are fall under their own licenses. By using this repository, you must also comply with the terms of these external licenses.
## Citing
If you find HaWoR useful for your research, please consider citing our paper:

```bibtex
@article{zhang2025hawor,
      title={HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos},
      author={Zhang, Jinglei and Deng, Jiankang and Ma, Chao and Potamias, Rolandos Alexandros},
      journal={arXiv preprint arXiv:2501.02973},
      year={2025}
    }
```
