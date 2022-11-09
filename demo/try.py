# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

def grab_frame(cap):
    ret,frame = cap.read()
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)



    
def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    config_file="C:\Code\detectron2-windows\configs\quick_schedules\mask_rcnn_R_50_FPN_inference_acc_test.yaml"
    cfg.merge_from_file(config_file)
    #cfg.merge_from_file(args.config_file)
    cfg.merge_from_list([])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: ")

    cfg = setup_cfg()

    demo = VisualizationDemo(cfg)
    cam = cv2.VideoCapture(0)

    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(grab_frame(cam))
    plt.ion()

    for vis in tqdm.tqdm(demo.run_on_video(cam)):
        #print(vis.type())
        #cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        #cv2.imshow(WINDOW_NAME, vis)
        #if cv2.waitKey(1) == 27:
        #    break  # esc to quit



        #plt.imshow(vis)
        #plt.show()

        im1.set_data(vis)
    cam.release()
    cv2.destroyAllWindows()


