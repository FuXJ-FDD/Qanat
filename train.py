import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model=r'C:\Deeplearning\YOLO\YOLOv11\ultralytics\cfg\models\11\yolo11n.yaml')
    model.train(data=r'datasets/train/250823_no/250823_no.yaml',
                imgsz=1280,
                epochs=300,
                patience=50,
                batch=4,
                workers=0,
                device='',
                optimizer='SGD',
                #close_mosaic=10,
                resume=False,
                project='runs/train',
                name='qanat_zong_yolov11n_5070_no',
                single_cls=False,
                cache=False,
                amp=False,
                )
