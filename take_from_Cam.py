import cv2
import argparse
from pathlib import Path
from PIL import Image
from mtcnn import MTCNN
from datetime import datetime

from PIL import Image
import numpy as np
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name','-n', default='unknown', type=str,help='input the name of the recording person')
args = parser.parse_args()
from pathlib import Path
data_path = Path('data')
save_path = data_path/'facebank'/args.name
if not save_path.exists():
    save_path.mkdir()

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
mtcnn = MTCNN()

while cap.isOpened():
    isSuccess,frame = cap.read()
    if isSuccess:
        # 实现按下“t”键拍照
        p =  Image.fromarray(frame[...,::-1])
        boxes ,faces = mtcnn.align_multi(p)

        if len(boxes)>0:
            scores = []
            for box in boxes:
                scores.append(box[4])
            scores = np.array(scores)
            best_face_index = np.argmax(scores)
            best_face_box = boxes[best_face_index]
            cv2.rectangle(frame, (int(best_face_box[0]),int(best_face_box[1])), (int(best_face_box[2]),int(best_face_box[3])), (0,0,255), 3)
            warped_face = faces[best_face_index]
            warped_face = np.array(warped_face)[...,::-1]
            
        cv2.imshow("Capture",frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            try:            
                cv2.imwrite(str(save_path/'{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-"))), warped_face)
            except:
                print('no face captured')
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
