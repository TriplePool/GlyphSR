import torch
import sys
import time
import os
from time import gmtime, strftime
from datetime import datetime
from tqdm import tqdm
import math
import copy

sys.path.append('../../')
sys.path.append('../')
from utils import util, ssim_psnr
from interfaces import base_GlyphSR as base
from utils.metrics import get_string_aster, get_string_crnn, Accuracy
from utils.util import str_filt

ssim = ssim_psnr.SSIM()

class TextSR(base.TextBase):

    def test(self):
        self.global_img_val_cnt = 0
        TP_Generator_dict = {
            "CRNN": self.CRNN_init,
            "OPT": self.TPG_init,
            "PARSeq": self.TPG_init_parseq,
        }

        tpg_opt = self.opt_TPG

        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init(0)
        model = model_dict['model']

        model_list = [model]

        aster, aster_info = TP_Generator_dict[self.args.tpg](recognizer_path=None,
                                                             opt=tpg_opt)  # self.args.tpg default=CRNN ，it is same as CRNN_init()

        test_bible = {}

        if self.args.test_model == "CRNN":
            crnn, aster_info = self.TPG_init(recognizer_path=None, opt=tpg_opt) if self.args.CHNSR else self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                'model': crnn,
                'data_in_fn': self.parse_crnn_data,
                'string_process': get_string_crnn
            }

        elif self.args.test_model == "ASTER":
            aster_real, aster_real_info = self.Aster_init()
            aster_info = aster_real_info
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }

        elif self.args.test_model == "MORAN":
            moran = self.MORAN_init()
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }

        aster_student = []

        recognizer_path = os.path.join(self.resume, "recognizer_best_acc_0.pth")
        print("recognizer_path:", recognizer_path)
        if os.path.isfile(recognizer_path):
            aster_student_, aster_stu_info = TP_Generator_dict[self.args.tpg](recognizer_path=recognizer_path,
                                                                              opt=tpg_opt)  #
        else:
            aster_student_, aster_stu_info = TP_Generator_dict[self.args.tpg](recognizer_path=None, opt=tpg_opt)

        if type(aster_student_) == list:
            aster_student_ = aster_student_[0]

        aster_student.append(aster_student_)

        aster.eval()
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))

        print('======================================================')
        current_acc_dict = {}
        current_fps_dict = {}
        current_psnr_dict = {}
        current_ssim_dict = {}

        for k, val_loader in enumerate(val_loader_list):
            data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
            print('evaling %s' % data_name)
            for model in model_list:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

            for stu in aster_student:
                stu.eval()
                for p in stu.parameters():
                    p.requires_grad = False

            metrics_dict = self.eval(
                model_list,
                val_loader,
                [test_bible[self.args.test_model], aster_student, aster],  #
                aster_info
            )

            acc = metrics_dict['accuracy']
            current_acc_dict[data_name] = float(acc)
            current_fps_dict[data_name] = float(metrics_dict['fps'])
            current_psnr_dict[data_name] = float(metrics_dict['psnr_avg'])
            current_ssim_dict[data_name] = float(metrics_dict['ssim_avg'])

            if acc > best_history_acc[data_name]:
                best_history_acc[data_name] = float(acc)
                print('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

            else:
                print('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))

        print('fps {:,.4f} | acc_avg {:.4f} | psnr_avg {:.4f} | ssim_avg {:.4f}\t'
              .format(sum(current_fps_dict.values()) / 3, sum(current_acc_dict.values()) / 3,
                      sum(current_psnr_dict.values()) / 3, sum(current_ssim_dict.values()) / 3, ))

    def eval(self, model_list, val_loader, aster, aster_info):
        n_correct = 0
        n_correct_lr = 0
        n_correct_hr = 0
        sum_images = 0
        metric_dict = {'psnr_lr': [], 'ssim_lr': [], 'cnt_psnr_lr': [], 'cnt_ssim_lr': [], 'psnr': [], 'ssim': [],
                       'cnt_psnr': [], 'cnt_ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'fps': 0.0}
        counter = 0

        sr_infer_time = 0.0

        for i, data in (enumerate(val_loader)):
            time_begin = time.time()

            images_hr, images_lr, label_strs, label_vecs_gt = data

            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)

            cascade_images = images_lr

            stu_model = aster[1][0]
            aster_dict_lr = self.parse_data_dict[self.args.tpg](cascade_images[:, :3, :, :])  #
            label_vecs_logits = stu_model(aster_dict_lr)  # .transpose(0,1)
            label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
            if self.args.tpg == 'CRNN':
                label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
            else:
                label_vecs_final = label_vecs.permute(0, 2, 1).unsqueeze(2)

            cascade_images, rec_out = model_list[0](images_lr, label_vecs_final)
            images_sr=cascade_images

            time_end = time.time()
            tmp = time_end - time_begin
            sr_infer_time += tmp

            if self.args.test_model == 'CRNN':
                aster_dict_lr = aster[0]["data_in_fn"](images_lr)  # [:, :3, ...]
                aster_dict_hr = aster[0]["data_in_fn"](images_hr)  # [:, :3, ...]
            else:
                aster_dict_lr = aster[0]["data_in_fn"](images_lr[:, :3, ...])  # [:, :3, ...]
                aster_dict_hr = aster[0]["data_in_fn"](images_hr[:, :3, ...])  # [:, :3, ...]

            if self.args.test_model == "MORAN":
                # LR
                aster_output_lr = aster[0]["model"](
                    aster_dict_lr[0],
                    aster_dict_lr[1],
                    aster_dict_lr[2],
                    aster_dict_lr[3],
                    test=True,
                    debug=True
                )
                # HR
                aster_output_hr = aster[0]["model"](
                    aster_dict_hr[0],
                    aster_dict_hr[1],
                    aster_dict_hr[2],
                    aster_dict_hr[3],
                    test=True,
                    debug=True
                )
            else:
                aster_output_lr = aster[0]["model"](aster_dict_lr)
                aster_output_hr = aster[0]["model"](aster_dict_hr)

            '-----------------------get pred of sr------------------------------------'


        # for i in range(self.args.stu_iter):
            image = images_sr
            if self.args.test_model == "CRNN":
                aster_dict_sr = aster[0]["data_in_fn"](image)
            else:
                aster_dict_sr = aster[0]["data_in_fn"](image[:, :3, :, :])
            # aster_dict_sr = aster[0]["data_in_fn"](image)
            if self.args.test_model == "MORAN":
                # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                aster_output_sr = aster[0]["model"](
                    aster_dict_sr[0],
                    aster_dict_sr[1],
                    aster_dict_sr[2],
                    aster_dict_sr[3],
                    test=True,
                    debug=True
                )
            else:
                aster_output_sr = aster[0]["model"](aster_dict_sr)
            # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
            if self.args.test_model == "CRNN":
                predict_result_sr = aster[0]["string_process"](aster_output_sr, self.args.CHNSR)
            elif self.args.test_model == "ASTER":
                predict_result_sr, _ = aster[0]["string_process"](
                    aster_output_sr['output']['pred_rec'],
                    aster_dict_sr['rec_targets'],
                    dataset=aster_info
                )
            elif self.args.test_model == "MORAN":
                preds, preds_reverse = aster_output_sr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                if type(sim_preds) == list:
                    predict_result_sr = [pred.split('$')[0] for pred in sim_preds]
                else:
                    predict_result_sr = [sim_preds.split('$')[0]]  # [pred.split('$')[0] for pred in sim_preds]

            img_sr = images_sr
            img_hr = images_hr

            metric_dict['psnr'].append(self.cal_psnr(img_sr, img_hr))
            metric_dict['ssim'].append(self.cal_ssim(img_sr, img_hr))

            if self.args.test_model == "CRNN":
                predict_result_lr = aster[0]["string_process"](aster_output_lr, self.args.CHNSR)
                predict_result_hr = aster[0]["string_process"](aster_output_hr, self.args.CHNSR)
            elif self.args.test_model == "ASTER":
                predict_result_lr, _ = aster[0]["string_process"](
                    aster_output_lr['output']['pred_rec'],
                    aster_dict_lr['rec_targets'],
                    dataset=aster_info
                )
                predict_result_hr, _ = aster[0]["string_process"](
                    aster_output_hr['output']['pred_rec'],
                    aster_dict_hr['rec_targets'],
                    dataset=aster_info
                )
            elif self.args.test_model == "MORAN":
                ### LR ###
                preds, preds_reverse = aster_output_lr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_lr[1].data)

                if type(sim_preds) == list:
                    predict_result_lr = [pred.split('$')[0] for pred in sim_preds]
                else:
                    predict_result_lr = [sim_preds.split('$')[0]]

                ### HR ###
                preds, preds_reverse = aster_output_hr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_hr[1].data)

                if type(sim_preds) == list:
                    predict_result_hr = [pred.split('$')[0] for pred in sim_preds]
                else:
                    predict_result_hr = [sim_preds.split('$')[0]]

            for batch_i in range(len(images_lr)):
                label = label_strs[batch_i]
                if predict_result_sr[batch_i] == str_filt(label, 'lower'):
                    counter += 1

                if predict_result_lr[batch_i] == str_filt(label, 'lower'):
                    n_correct_lr += 1
                if predict_result_hr[batch_i] == str_filt(label, 'lower'):
                    n_correct_hr += 1
                self.global_img_val_cnt += 1

            sum_images += val_batch_size

            torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / (len(metric_dict['psnr']) + 1e-10)
        ssim_avg = sum(metric_dict['ssim']) / (len(metric_dict['ssim']) + 1e-10)
        fps = sum_images / sr_infer_time

        print('[{}]\t'

              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg), float(ssim_avg)))
        metric_dict['fps'] = fps

        acc = round(counter / sum_images, 4)

        accuracy_lr = round(n_correct_lr / sum_images, 4)
        accuracy_hr = round(n_correct_hr / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)


        print('sr_accuray: %.2f%%' % (acc * 100))
        accuracy = acc

        print('lr_accuray: %.2f%%' % (accuracy_lr * 100))
        print('hr_accuray: %.2f%%' % (accuracy_hr * 100))

        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg

        return metric_dict
