import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from collections import deque
import numpy as np
import copy


label_dict = {}
id_lst = []
counter = 0
(dX, dY) = (0, 0)

def inter_from_256(x):
    return np.interp(x=x,xp=[0,255],fp=[0,1])

def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')


    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)


    device = select_device(opt.device)
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    half = device.type != 'cpu'  
    model = attempt_load(yolo_weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names 
    if half:
        model.half()  # to FP16

    vid_path, vid_writer = None, None
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    heatMapArray = []
    iter = 1

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference 
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            #im0 = cv2.resize(im0, (0,0), fx=0.5, fy=0.5)
            h_img, w_img, channels = im0.shape[0], im0.shape[1], im0.shape[2]
            if iter == 1:
                iter = 0
                heatMapArray = np.ones([h_img, w_img, channels], dtype=np.uint8)

            s += '%gx%g ' % img.shape[2:]  # print string
            # save_path = str(Path(out) / Path(p).name)
            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        x2, x1, y2, y1 = int(bboxes[2].item()), int(bboxes[0].item()), int(bboxes[3].item()), int(bboxes[1].item())
                        cx, cy = annotator.box_label(bboxes, label, color=colors(c, True))
                        srnk = 5
                        x1, y1 = cx-srnk, cy-srnk
                        x2, y2 = cx+srnk, cy+srnk

                        heatMapArray[y1:y2, x1:x2, :] += 1

                    ################## Heat Map #######################
                    heatMapArray = cv2.GaussianBlur(heatMapArray,(9,9),cv2.BORDER_DEFAULT)
                    cumalitive_intensity = cv2.normalize(heatMapArray, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap = cv2.applyColorMap(cumalitive_intensity.astype(np.uint8), cv2.COLORMAP_TURBO)
                    navy_lo=np.array([59,18, 48])
                    navy_hi=np.array([60,20,50])
                    mask = cv2.inRange(heatmap,navy_lo,navy_hi)
                    heatmap[mask>0]=(0,0,0)
                    overlay = cv2.addWeighted(im0, 1, heatmap, 0.5, 0)
                    final_img = cv2.hconcat([heatmap, overlay])

                    # save_path = "output_video"
                    # if vid_path != save_path:  # new video
                    #     vid_path = save_path
                    #     if isinstance(vid_writer, cv2.VideoWriter):
                    #         vid_writer.release()  # release previous video writer
                    #     else:  # stream
                    #         # fps, w, h = 3, im0.shape[1], im0.shape[0]
                    #         fps, w, h = 3, final_img.shape[1], final_img.shape[0]
                    #         save_path += '.avi'

                    #     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
                    
                    # vid_writer.write(final_img)
                    cv2.imshow(p, final_img)
    
            else:
                deepsort.increment_ages()

            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # cv2.imshow(p, heatmap)
            if cv2.waitKey(25) == ord('q'):  # q to quit
                exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
