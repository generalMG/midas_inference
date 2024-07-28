import cv2 
import torch 
import matplotlib.pyplot as plt 
from enum import Enum
import os 
import numpy as np

class ModelType(Enum):
    DPT_LARGE = "DPT_Large"
    DPT_Hybrid = "DPT_Hybrid"
    MIDAS_SMALL = "MiDaS_small"

class Midas():
    def __init__(self,modelType:ModelType=ModelType.DPT_LARGE): 
        self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
        self.modelType = modelType

    def load_device(self):
        try:
            self.device = torch.device("mps")
            print("Loading to Apple MPS.")
        except:
            print("Could not load to Apple MPS.")
            if torch.cuda.is_available():
                print("Loading to CUDA device.")
                self.device = torch.device("cuda:0")
            else:
                print("Could not use CUDA, switching to CPU")
                self.device = torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
    
    def transform(self):
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.modelType.value == "DPT_Large" or self.modelType.value == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
    
    def prediction(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = self.transform(frame).to(self.device)
        
        with torch.no_grad():
            preds = self.midas(input_frame)
            preds = torch.nn.functional.interpolate(
                preds.unsqueeze(1),
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        
        depthMap = preds.cpu().numpy()
        depthMap = cv2.normalize(depthMap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depthMap = cv2.applyColorMap(depthMap, cv2.COLORMAP_INFERNO)
        return depthMap
    
    def livePredictions(self):
        capture = cv2.VideoCapture(1)
        while True:
            ret, frame = capture.read()

            if not ret or frame is None:
                continue

            depthMap = self.prediction(frame)
            combined = np.hstack((frame, depthMap))
            cv2.imshow("output", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

def run(modelType: ModelType):
    midas = Midas(modelType)
    midas.load_device()
    midas.transform()
    midas.livePredictions()

run(ModelType.MIDAS_SMALL)