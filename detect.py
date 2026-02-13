# -*- coding: utf-8 -*-
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'C:\Deeplearning\YOLO\YOLOv11\runs\train\qanat_zong_yolov11n_5070\weights\best.pt')
    model.predict(source=r'D:\坎儿井2期-全伊朗计划-一部\input\main1_00\01',
                  imgsz=1280,
                  project=r'D:\坎儿井2期-全伊朗计划-一部\output\main1_00',
                  name='01',
                  conf=0.402,
                  save=True,
                  show=False,
                  show_labels=True,
                  show_conf=False,
                  save_txt=True,
                  )
