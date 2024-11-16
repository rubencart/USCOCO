# Explicitly Representing Syntax Improves Sentence-to-layout Prediction of Unexpected Situations

Code, data and checkpoints for our [paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00643/120575) 
published in TACL.
Version on arXiv contains minor corrections (typo's etc): [link](https://arxiv.org/abs/2401.14212).

## Install

Dependencies:
- python 3.8
- pytorch 1.12
- transformers 4.28
- pytorch-lightning 1.9
- wandb
- typed-argument-parser
- h5py
- seaborn
- iglovikov_helper_functions
- pycocotools
- albumentations
- clip
- nltk
- pyyaml

```
conda create -n uscocoenv python=3.8 h5py -c conda-forge
conda activate uscocoenv
conda install pytorch=1.12 torchvision torchaudio cudatoolkit=11.3 -c pytorch
# conda install h5py
pip install pytorch-lightning==1.9 wandb transformers==4.28 typed-argument-parser seaborn iglovikov_helper_functions pycocotools albumentations nltk pyyaml
pip install git+https://github.com/openai/CLIP.git
```

## Run

### Training

Edit or copy a `.yaml` file in `config/` and change the settings, or change the default settings in `config.py`
(at least change the paths marked with `# todo` in `config.py` or set them in a `.yaml`).

Set `train_image_dir`, `val_image_dir`, `train_captions`, `val_captions`, `train_instances`, `val_instances` 
to where you downloaded the COCO dataset (images + captions and instances json's, all in the 2017 version).
Unzip all the `.zip` files in `data/`.
Run with:
```
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/tg.yaml --seed 42
```

Checkpoints of pretrained text encoders are downloaded from the Huggingface hub. 
GPT-2 checkpoints were already available, TG, PLM, PLM_mask and AttnGAN checkpoints have been uploaded to
https://huggingface.co/rubencart.
These can be automatically downloaded when starting a run with the following settings (already set in the `.yaml` files
in `config/` where necessary).
```
text_encoder:
 download_from_hub: True
 hub_path: "rubencart/TG_sm_text_encoder"
 # hub_path: "rubencart/GPT-2_Bllip_sm_text_encoder"
 # hub_path: ...
```

### Use trained models for inference

Coming soon...

## Replicate experiments

### Preprocessing

```
# GPT-2_bllip with all preprocessing
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_bllip.yaml --seed 42
# GPT-2_bllip with no preprocessing
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_bllip_no_CPN_no_SP.yaml --seed 42
# GPT-2_bllip with only SP (no CPN)
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_bllip_no_CPN.yaml --seed 42
# GPT-2_bllip with only CPN (no SP)
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_bllip_no_SP.yaml --seed 42

# PLM with all preprocessing
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/plm.yaml --seed 42
# PLM with no preprocessing
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/plm_no_CPN_no_SP.yaml --seed 42
# PLM with only SP (no CPN)
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/plm_no_CPN.yaml --seed 42
# PLM with only CPN (no SP)
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/plm_no_SP.yaml --seed 42
```


### Table 2 (layout predictor comparison)

In order of appearance:

```
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/table_2/obj_lstm_attn_gan.yaml --seed 42

CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/table_2/obj_lstm_gpt2_bllip.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/table_2/obj_lstm_lrg_gpt2_bllip.yaml --seed 42

CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/table_2/seq_gpt2_bllip.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/table_2/par_gpt2_bllip.yaml --seed 42

CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/table_2/seq_tg.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/table_2/par_tg.yaml --seed 42
```

### Table 3 (text encoder comparison with and without structural loss)

In order of appearance:

```
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_Lstruct.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_bllip.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_bllip_Lstruct.yaml --seed 42

CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_lrg.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_lrg_Lstruct.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_bllip_lrg.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_bllip_lrg_Lstruct.yaml --seed 42

CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/llama_7B.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/llama_7B_Lstruct.yaml --seed 42

CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/plm.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/plm_Lstruct.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/plm_mask.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/plm_mask_Lstruct.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/tg.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/tg_Lstruct.yaml --seed 42

CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/tg_lrg.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/tg_lrg_Lstruct.yaml --seed 42
```

### Table 4 (Final model results)

In order of appearance:

```
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_bllip_shuffle.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_bllip.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_lrg.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/gpt2_bllip_lrg.yaml --seed 42

CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/llama_7B.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/llama_30B.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/clip.yaml --seed 42  # not in table

CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/tg_rightbranch.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/tg_rightbranch_Lstruct.yaml --seed 42

CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/plm_Lstruct.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/plm_mask_Lstruct.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/tg_Lstruct.yaml --seed 42
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/explicit/tg_lrg_Lstruct.yaml --seed 42
```

### Probe

Run first with (in `.yaml`):
```
save_probe_embeddings: True
save_probe_embeddings_train: True
```

E.g.:
```
CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/generate_probe_emb.yaml --seed 43
```

And then start a probe run:
```
CUDA_VISIBLE_DEVICES=0 python src/probe/main.py --cfg config/probe/gpt2_bin_allneg.yaml --seed 43
```

### Train TG models

As mentioned in the article, we train TG models based on the paper by Sartran et al., but with minor changes,
and with our own implementation.
We based this implementation on Qian et al.'s implementation of their PLM/PLM_mask models, and provide our code in a fork 
of their repo: [https://github.com/rubencart/transformers-struct-guidance](https://github.com/rubencart/transformers-struct-guidance).

## USCOCO data

Collected captions and bounding boxes for unexpected situations 
are included in COCO captions/detection format, in files
`./data/USCOCO_captions.json` and `./data/USCOCO_instances.json`.

## Checkpoints

Coming soon...

## Citation

If you use our work, please use the following reference.
```
@article{nuyts2024explicitly,
  title={Explicitly Representing Syntax Improves Sentence-to-Layout Prediction of Unexpected Situations},
  author={Nuyts, Wolf and Cartuyvels, Ruben and Moens, Marie-Francine},
  journal={Transactions of the Association for Computational Linguistics},
  volume={12},
  pages={264--282},
  year={2024},
  publisher={MIT Press One Broadway, 12th Floor, Cambridge, Massachusetts 02142, USA}
}
```
