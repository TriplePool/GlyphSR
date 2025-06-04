import torch
import sys
import os
import string
from collections import OrderedDict

import ptflops


from model import recognizer
from model import moran
from model import crnn
from dataset import lmdbDataset_real, alignCollate_realWTL
sys.path.append('../../')
from utils import util, ssim_psnr, utils_moran, utils_crnn
from utils.labelmaps import get_vocabulary, labels2strs

class TextBase(object):
    def __init__(self, config, args, opt_TPG=None):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.scale_factor = self.config.TRAIN.down_sample_scale
        self.opt_TPG = opt_TPG
        #数据集加载的类
        self.align_collate = alignCollate_realWTL
        self.load_dataset = lmdbDataset_real
        self.vis = args.vis

        self.align_collate_val = alignCollate_realWTL
        self.load_dataset_val = lmdbDataset_real#Distorted# BadSet


        self.resume = args.resume if args.resume is not None else config.TRAIN.resume
        self.batch_size = args.batch_size if args.batch_size is not None else self.config.TRAIN.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation,
            'chinese': open("al_chinese.txt", "r").readlines()[0].replace("\n", "")
        }
        self.test_data_dir = self.args.test_data_dir if self.args.test_data_dir is not None else self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.max_len = config.TRAIN.max_len
        self.vis_dir = self.args.vis_dir if self.args.vis_dir is not None else self.config.TRAIN.VAL.vis_dir
        self.ckpt_path = os.path.join('ckpt', self.vis_dir)
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.cal_psnr_weighted = ssim_psnr.weighted_calculate_psnr
        self.cal_ssim_weighted = ssim_psnr.SSIM_WEIGHTED()
        self.mask = self.args.mask
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)

        self.parse_data_dict = {
            "CRNN": self.parse_crnn_data,
            "OPT": self.parse_crnn_data,
            "PARSeq": self.parse_parseq_data,
        }

    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    def get_test_data(self, dir_):
        cfg = self.config.TRAIN
        self.args.test_data_dir

        test_dataset = self.load_dataset_val(root=dir_,
                                         voc_type=cfg.voc_type,
                                         max_len=cfg.max_len,
                                         test=True,
                                         )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate_val(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask, train=False),
            drop_last=False)
        return test_dataset, test_loader

    def generator_init(self, iter=0, resume_in=None):
        cfg = self.config.TRAIN

        resume = self.resume
        if not resume_in is None:
            resume = resume_in

        if self.args.noGFM:
            from model import GlyphSR_no_GFM
            model = GlyphSR_no_GFM.GlyphSR(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                hidden_units=self.args.hd_u,
                                text_emb=95 if self.args.tpg == 'PARSeq' else 37)
        else:
            from model import GlyphSR
            model = GlyphSR.GlyphSR(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                hidden_units=self.args.hd_u,
                                text_emb=95 if self.args.tpg == 'PARSeq' else 37,
                                char_block=self.args.char_block)


        channel_size = 3 if self.args.arch in ["srcnn", "edsr", "vdsr", ""] else 4
        macs, params = ptflops.get_model_complexity_info(model, (channel_size, cfg.height//cfg.down_sample_scale, cfg.width//cfg.down_sample_scale), as_strings=True,
                                                         print_per_layer_stat=False, verbose=True)
        print("---------------- SR Module -----------------")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("--------------------------------------------")


        model = model.to(self.device)
        if not resume == '':
            print('loading pre-trained model from %s ' % resume)
            if self.config.TRAIN.ngpu == 1:
                # if is dir, we need to initialize the model list
                if os.path.isdir(resume):
                    print("resume:", resume)
                    model_dict = torch.load(
                            os.path.join(resume, "model_best_acc_" + str(iter) + ".pth")
                        )['state_dict_G']
                    model.load_state_dict(
                        model_dict
                    , strict=True)
                else:
                    loaded_state_dict = torch.load(resume)
                    if 'state_dict_G' in loaded_state_dict:
                        model.load_state_dict(torch.load(resume)['state_dict_G'])
                    else:
                        model.load_state_dict(torch.load(resume))
            else:
                model_dict = torch.load(
                    os.path.join(resume, "model_best_acc_" + str(iter) + ".pth")
                )['state_dict_G']

                if os.path.isdir(resume):
                    model.load_state_dict(
                        {'module.' + k: v for k, v in model_dict.items()}
                        , strict=False)
                else:
                    model.load_state_dict(
                    {'module.' + k: v for k, v in torch.load(resume)['state_dict_G'].items()})
        return {'model': model}

    def MORAN_init(self):
        cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = self.config.TRAIN.VAL.moran_pretrained
        print('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        MORAN = torch.nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]

        in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100

        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, in_width), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def CRNN_init(self, recognizer_path=None, opt=None):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)

        macs, params = ptflops.get_model_complexity_info(model, (1, 32, 100), as_strings=True,
                                                         print_per_layer_stat=False, verbose=True)
        print("---------------- TP Module -----------------")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("--------------------------------------------")

        print("recognizer_path:", recognizer_path)

        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        stat_dict = torch.load(model_path)
        # print("stat_dict:", stat_dict.keys())
        if recognizer_path is None:
            model.load_state_dict(stat_dict)
        else:
            # print("stat_dict:", stat_dict)
            # print("stat_dict:", type(stat_dict) == OrderedDict)
            if type(stat_dict) == OrderedDict:
                print("The dict:")
                model.load_state_dict(stat_dict)
            else:
                print("The model:")
                model = stat_dict
        # model #.eval()
        # model.eval()
        return model, aster_info

    def CRNNRes18_init(self, recognizer_path=None, opt=None):
        model = crnn.CRNN_ResNet18(32, 1, 37, 256)
        model = model.to(self.device)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        stat_dict = torch.load(model_path)
        # print("stat_dict:", stat_dict.keys())
        if recognizer_path is None:
            if stat_dict == model.state_dict():
                model.load_state_dict(stat_dict)
        else:
            model = stat_dict
        # model #.eval()
        # model.eval()
        return model, aster_info

    def TPG_init(self, recognizer_path=None, opt=None):
        model = crnn.Model(opt)
        model = model.to(self.device)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else opt.saved_model
        print('loading pretrained TPG model from %s' % model_path)
        stat_dict = torch.load(model_path)

        model_keys = model.state_dict().keys()
        #print("state_dict:", len(stat_dict))
        if type(stat_dict) == list:
            print("state_dict:", len(stat_dict))
            stat_dict = stat_dict[0]#.state_dict()
        #load_keys = stat_dict.keys()

        # print("recognizer_path:", recognizer_path)
        if recognizer_path is None:
            # model.load_state_dict(stat_dict)
            load_keys = stat_dict.keys()
            man_load_dict = model.state_dict()
            for key in stat_dict:
                if not key.replace("module.", "") in man_load_dict:
                    print("Key not match", key, key.replace("module.", ""))
                man_load_dict[key.replace("module.", "")] = stat_dict[key]
            model.load_state_dict(man_load_dict)
        else:
            #model = stat_dict
            model.load_state_dict(stat_dict)

        return model, aster_info
    
    def TPG_init_parseq(self, recognizer_path=None, opt=None):
        model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
        # model = torch.hub.load('/home/path/.cache/torch/hub/baudm_parseq_main/', 'parseq', pretrained=True,
        #                        source='local')

        model = model.to(self.device)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else 'baudm/parseq'
        print('loading pretrained parseq model from %s' % model_path)
        if recognizer_path is not None:
            model_path = recognizer_path
            stat_dict = torch.load(model_path)
            # print("stat_dict:", stat_dict.keys())
            if type(stat_dict) == OrderedDict:
                    print("The dict:")
                    model.load_state_dict(stat_dict)
            else:
                print("The model:")
                model = stat_dict
        
        return model, aster_info
    
    def parse_parseq_data(self, imgs_input_, ratio_keep=False):
        # in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100
        in_width = 128

        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

        # print("imgs_input:", imgs_input.shape)

        tensor = imgs_input * 2 - 1
        return tensor

    def parse_crnn_data(self, imgs_input_, ratio_keep=False):

        in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100

        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

        # print("imgs_input:", imgs_input.shape)

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def parse_OPT_data(self, imgs_input_, ratio_keep=False):

        in_width = 512

        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

        # print("imgs_input:", imgs_input.shape)

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def Aster_init(self):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        aster.load_state_dict(torch.load(self.config.TRAIN.VAL.rec_pretrained)['state_dict'])
        print('load pred_trained aster model from %s' % self.config.TRAIN.VAL.rec_pretrained)
        aster = aster.to(self.device)
        aster = torch.nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        aster.eval()
        return aster, aster_info

    def parse_aster_data(self, imgs_input):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all', 'chinese']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)
