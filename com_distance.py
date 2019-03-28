import cv2
from PIL import Image
import argparse
from pathlib import Path
import numpy as np
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from model import l2_norm
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    learner.load_state(conf, 'ir_se50.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
#    targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
#    print('facebank updated')

#  face 1
    frame = cv2.imread("/home/zzg/DeepLearning/InsightFace_Pytorch/data/facebank/wang/2019-03-10-18-26-57.jpg")
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
        warped_face1 = faces[best_face_index]
#        warped_face1 = np.array(warped_face)[...,::-1]
    else:
        print("NO face detect in Image1")

#### face 2

    frame = cv2.imread("/home/zzg/DeepLearning/InsightFace_Pytorch/data/facebank/wang/2019-03-10-18-27-06.jpg")
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
        warped_face2 = faces[best_face_index]
#        warped_face2 = np.array(warped_face)[...,::-1]
    else:
        print("NO face detect in Image2")

    embed1 = learner.model(conf.test_transform(warped_face1).to(conf.device).unsqueeze(0))
    embed2 = learner.model(conf.test_transform(warped_face2).to(conf.device).unsqueeze(0))
    embed1 = l2_norm(embed1)
    embed2 = l2_norm(embed2)
    
    pdist = torch.nn.PairwiseDistance(p=2)
    dist1 = pdist(embed1,embed2)
    dist1 = dist1.detach().cpu()
    
    
    embed1 = embed1.detach().cpu().numpy()
    embed2 = embed2.detach().cpu().numpy()
    
    dist = np.sqrt(np.sum(np.square(embed1 - embed2)))
    print(dist)
    print(dist1)

