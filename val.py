import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/qanat_zong_yolov11s_1660Ti/weights/best.pt')
    model.val(data='datasets/val/zong-qanat-google/zong-qanat-google.yaml',
              split='val',
              imgsz=1280,
              batch=2,
              iou=0.9,
              rect=False,
              save_json=False,
              project='runs/val',
              name='zong-qanat-google-yolov11s_iou0.9',
              )