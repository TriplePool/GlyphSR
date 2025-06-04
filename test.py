import yaml
import sys
import argparse
import os
from IPython import embed
from easydict import EasyDict
from interfaces.super_resolution_GlyphSR import TextSR
os.chdir(sys.path[0])
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 


def main(config, args, opt_TPG):
    Mission = TextSR(config, args, opt_TPG)

    Mission.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='GlyphSR',)
    parser.add_argument('--noGFM', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='./TextZoom/test/medium/', help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--test_model', type=str, default='CRNN', choices=['ASTER', "CRNN", "MORAN"])
    parser.add_argument('--tpg', type=str, default="CRNN", choices=['CRNN', 'OPT', 'PARSeq'])
    parser.add_argument('--config', type=str, default='config/super_resolution_fam_simple_rec_char.yaml')
    parser.add_argument('--CHNSR', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--char_block', type=int, default=6)

    args = parser.parse_args()
    # config_path = os.path.join('config', args.config)
    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)

    opt = {
        "Transformation": 'None',
        "FeatureExtraction": 'ResNet',
        "SequenceModeling": 'None',
        "Prediction": 'CTC',
        "num_fiducial": 20,
        "input_channel": 1,
        "output_channel": 512,
        "hidden_size": 256,
        "saved_model": "best_accuracy.pth",#"best_accuracy.pth", #"None-ResNet-None-CTC.pth",#"CRNN-PyTorchCTC.pth", # None-ResNet-None-CTC.pth
        "character": "-0123456789abcdefghijklmnopqrstuvwxyz"
    }

    opt["num_class"] = len(opt['character'])
    if args.vis: #check save path exist ?
        os.makedirs(args.vis_dir,exist_ok=True)

    opt = EasyDict(opt)
    print(opt)
    print(args)
    print(config)
    main(config, args, opt_TPG=opt)
