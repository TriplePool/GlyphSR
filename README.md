# GlyphSR: Glyph-Aware Scene Text Image Super-Resolution

This repository provides the testing code for “GlyphSR: A Simple Glyph-Aware Framework for Scene Text Image Super-Resolution” (AAAI 2025). Use it to reproduce inference results on TextZoom with pretrained GlyphSR checkpoints.

---

## Setup
```
cd GlyphSR
source scripts/setup_env.sh
```
- Download the [TextZoom](https://drive.google.com/drive/folders/1WRVy-fC_KrembPkaI68uqQ9wyaptibMh?usp=sharing) dataset and pretrained weights of [CRNN](https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0)
- Change corresponding paths in the config file config/super_resolution_base.yaml 
- download [GlyphSR checkpoints](https://pan.baidu.com/s/1ojyLdUEz_l0otCdb4tjLqg?pwd=366b)

## Test for GlyphSR
```
CUDA_VISIBLE_DEVICES=1 python3 test.py --test_model="CRNN" --batch_size=48  --STN  --mask --resume=ckpt/glyphsr/ --tpg PARSeq --config config/super_resolution_base.yaml --char_block 5
```

## Test for GlyphSR w/o GFM
```
CUDA_VISIBLE_DEVICES=1 python3 test.py --test_model="CRNN" --batch_size=48  --STN  --mask --resume=ckpt/glyphsr_no_gfm/ --tpg PARSeq --config config/super_resolution_base.yaml --noGFM
```

## Citation
If you find this project useful in your research, please consider cite:
```bibtex
@inproceedings{wei2025glyphsr,
  title={GlyphSR: A Simple Glyph-Aware Framework for Scene Text Image Super-Resolution},
  author={Wei, Baole and Zhou, Yuxuan and Gao, Liangcai and Tang, Zhi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={8},
  pages={8277--8285},
  year={2025}
}
```