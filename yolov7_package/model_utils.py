import argparse
import colorsys
import os, sys
import pathlib

import yaml

import cv2
import torch
import numpy as np

import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch import optim

from .models.experimental import attempt_load
from .models.yolo import Model
from .utils.datasets import create_dataloader
from .utils.general import non_max_suppression, non_max_suppression_kpt, check_dataset, init_seeds, one_cycle, \
    check_img_size, colorstr
from .utils.torch_utils import TracedModel, torch_distributed_zero_first, ModelEMA
from .load_utils import load_script_model


def conf2color(conf):
    if (conf <= 0.6):
        color = colorsys.hsv_to_rgb(15 / 360.0, 1, 1)
    else:
        color = colorsys.hsv_to_rgb((25 + 100 * (2 * conf - 1.2)) / 360.0, 1, 1)
    return [k * 255 for k in reversed(color)]

coco_names = ["person",
              "bicycle",
              "car",
              "motorbike",
              "aeroplane",
              "bus",
              "train",
              "truck",
              "boat",
              "traffic light",
              "fire hydrant",
              "stop sign",
              "parking meter",
              "bench",
              "bird",
              "cat",
              "dog",
              "horse",
              "sheep",
              "cow",
              "elephant",
              "bear",
              "zebra",
              "giraffe",
              "backpack",
              "umbrella",
              "handbag",
              "tie",
              "suitcase",
              "frisbee",
              "skis",
              "snowboard",
              "sports ball",
              "kite",
              "baseball bat",
              "baseball glove",
              "skateboard",
              "surfboard",
              "tennis racket",
              "bottle",
              "wine glass",
              "cup",
              "fork",
              "knife",
              "spoon",
              "bowl",
              "banana",
              "apple",
              "sandwich",
              "orange",
              "broccoli",
              "carrot",
              "hot dog",
              "pizza",
              "donut",
              "cake",
              "chair",
              "sofa",
              "pottedplant",
              "bed",
              "diningtable",
              "toilet",
              "tvmonitor",
              "laptop",
              "mouse",
              "remote",
              "keyboard",
              "cell phone",
              "microwave",
              "oven",
              "toaster",
              "sink",
              "refrigerator",
              "book",
              "clock",
              "vase",
              "scissors",
              "teddy bear",
              "hair drier",
              "toothbrush"]



class Yolov7Detector:
    def __init__(self,
                 weights=None,
                 img_size=None,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 augment=False,
                 agnostic_nms=False,
                 device='cuda:0',
                 traced=False,
                 classes='coco'):
        if img_size is None:
            img_size = [640, 640]
        if weights is None and not traced:
            weights = 'yolov7.pt'
        elif weights is None:
            weights = 'yolov7.torchscript_16.pt'

        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        cpu = device.lower() == 'cpu'
        cuda = not cpu and torch.cuda.is_available()
        self.device = torch.device(device if cuda else 'cpu')

        self.half = self.device.type != 'cpu'

        sys.path.append(os.path.join(os.path.dirname(__file__), ""))

        if not traced:
            self.model = attempt_load([weights], map_location=self.device)  # load FP32 model
        else:
            if not os.path.isfile(weights):
                load_script_model(weights)
            self.model = torch.jit.load(weights, map_location=self.device).float().eval()

        self.traced = traced

        self.img_size = img_size  # check_img_size(img_size, s=self.stride)  # check img_size

        self.weights = weights
        #self.stride = int(self.model.stride.max())  # model stride

        if self.half:
            self.model.half()
        if classes == 'coco':
            self.names = coco_names
        else:
            with open(classes, 'r') as fp:
                self.names = fp.read().splitlines()
        self.model.eval()

    def detect(self, x) -> (list, list, list):
        """
        :param x: list of numpy images (e.g. after cv2.imread) or numpy image
        :return: []classes, []boxes, []confidences
        """
        if type(x) != list:
            x = [x]
        for i, img in enumerate(x):
            if type(img) != np.array:
                x[i] = np.array(img)
        with torch.no_grad():
            classes = []
            boxes = []
            confs = []
            img_sizes = []
            y = []
            for i, img in enumerate(x):
                img_sizes.append(img.shape)
                img_inf = cv2.resize(img, self.img_size)
                # img_inf = letterbox(img, self.img_size, stride=self.stride)[0]
                img_inf = img_inf[:, :, ::-1].transpose(2, 0, 1)
                img_inf = np.ascontiguousarray(img_inf)

                img_inf = torch.from_numpy(img_inf).to(self.device)
                img_inf = img_inf.half() if self.half else img_inf.float()  # uint8 to fp16/32
                img_inf /= 255.0  # 0 - 255 to 0.0 - 1.0
                y.append(img_inf)

            y = torch.stack(y)
            pred = self.model(y, )[0] # augment=self.augment

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None,
                                       agnostic=self.agnostic_nms)
            #print(pred)
            for i, det in enumerate(pred):  # detections per image
                local_classes = []
                local_boxes = []
                local_confs = []
                old_shape = img_sizes[i]
                #gn = torch.tensor(old_shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    dx = old_shape[0] / self.img_size[0]
                    dy = old_shape[1] / self.img_size[1]
                    #print(det[:, :4])
                    #det[:, :4] = scale_coords(self.img_size, det[:, :4], old_shape[:2]).round()
                    #print(det[:, :4])

                    for *xyxy, conf, cls in reversed(det):
                        coords = torch.tensor(xyxy).tolist()
                        xyxy_scaled = [coords[0] * dy, coords[1] * dx, coords[2] * dy, coords[3] * dx]
                        #print(, conf, cls)
                        #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        local_classes.append(int(cls.cpu().item()))
                        local_boxes.append(xyxy_scaled)
                        local_confs.append(float(conf.cpu().item()))

                classes.append(local_classes)
                boxes.append(local_boxes)
                confs.append(local_confs)


            return classes, boxes, confs

    def train(self, save_dir: str, data: str, cfg: str, hyp: str, transfer=False, workers=8, epochs=10, batch_size=32):
        if self.traced:
            raise ValueError('Can not train traced model!')
        save_dir = pathlib.Path(save_dir)
        with open(hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(hyp, f, sort_keys=False)
        init_seeds(1)
        with open(data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        is_coco = data.endswith('coco.yaml')
        model = Model(cfg, ch=3, nc=len(self.names), anchors=hyp.get('anchors')).to(self.device)
        with torch_distributed_zero_first(-1):
            check_dataset(data_dict)  # check

        train_path = data_dict['train']
        test_path = data_dict['val']

        freeze = [0]
        freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

            # Save run settings
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(hyp, f, sort_keys=False)
        total_batch_size = batch_size
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
            if hasattr(v, 'im'):
                if hasattr(v.im, 'implicit'):
                    pg0.append(v.im.implicit)
                else:
                    for iv in v.im:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imc'):
                if hasattr(v.imc, 'implicit'):
                    pg0.append(v.imc.implicit)
                else:
                    for iv in v.imc:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imb'):
                if hasattr(v.imb, 'implicit'):
                    pg0.append(v.imb.implicit)
                else:
                    for iv in v.imb:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imo'):
                if hasattr(v.imo, 'implicit'):
                    pg0.append(v.imo.implicit)
                else:
                    for iv in v.imo:
                        pg0.append(iv.implicit)
            if hasattr(v, 'ia'):
                if hasattr(v.ia, 'implicit'):
                    pg0.append(v.ia.implicit)
                else:
                    for iv in v.ia:
                        pg0.append(iv.implicit)
            if hasattr(v, 'attn'):
                if hasattr(v.attn, 'logit_scale'):
                    pg0.append(v.attn.logit_scale)
                if hasattr(v.attn, 'q_bias'):
                    pg0.append(v.attn.q_bias)
                if hasattr(v.attn, 'v_bias'):
                    pg0.append(v.attn.v_bias)
                if hasattr(v.attn, 'relative_position_bias_table'):
                    pg0.append(v.attn.relative_position_bias_table)
            if hasattr(v, 'rbr_dense'):
                if hasattr(v.rbr_dense, 'weight_rbr_origin'):
                    pg0.append(v.rbr_dense.weight_rbr_origin)
                if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
                    pg0.append(v.rbr_dense.weight_rbr_avg_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
                    pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
                    pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
                    pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
                if hasattr(v.rbr_dense, 'vector'):
                    pg0.append(v.rbr_dense.vector)

        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

        optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        ema = ModelEMA(model)

        # Resume
        start_epoch, best_fitness = 0, 0.0

        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz, imgsz_test = [check_img_size(x, gs) for x in self.img_size]  # verify imgsz are gs-multiples

        # DP mode
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, False,
                                                hyp=hyp, augment=True, cache=False, rect=False, rank=-1,
                                                world_size=1, workers=workers,
                                                image_weights=False, quad=False,
                                                prefix=colorstr('train: '))
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        nb = len(dataloader)  # number of batches
        assert mlc < len(self.names), 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (
        mlc, len(self.names), data, len(self.names) - 1)





    def draw_on_image(self, img, boxes: list, scores: list, class_ids: list, thickness=1):
        for i, box in enumerate(boxes):
            name = self.names[class_ids[i]]
            color = conf2color(scores[i])
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            img = cv2.rectangle(img, c1, c2, color, thickness=thickness)
            img = cv2.putText(img, name, (c1[0], c1[1] - 2), 0, 1, color, thickness=thickness)
        return img