# Explicitly Representing Syntax Improves Sentence-to-layout Prediction of Unexpected Situations

Code, data and checkpoints for our [paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00643/120575) 
published in TACL.

## Install

Dependencies:
- python 3.8
- pytorch 1.12
- transformers
- pytorch-lightning 1.6
- wandb
- typed-argument-parser
- h5py
- seaborn
- iglovikov_helper_functions
- pycocotools
- albumentations
- latent-structure-tools
- clip
- nltk
- pyyaml

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install h5py
pip install pytorch-lightning==1.6.1 wandb transformers typed-argument-parser seaborn iglovikov_helper_functions pycocotools albumentations nltk pyyaml
pip install git+https://github.com/openai/CLIP.git
```

## Run

Set `train_image_dir`, `val_image_dir`, `train_captions`, `val_captions`, `train_instances`, `val_instances` 
to where you downloaded the COCO dataset (images + captions and instances json's, all in the 2017 version).

```
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python src/main.py --cfg config/par/implicit/train_generate.yaml
```

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