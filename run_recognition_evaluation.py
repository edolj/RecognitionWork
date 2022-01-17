from cProfile import label
from calendar import c
from ctypes import resize
from tkinter import Image
import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist 
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()

        cla_d = self.get_annotations(self.annotations_path)
        
        # Change the following extractors, modify and add your own
        model = load_model('resnet50_best_model.h5')
        img_size = 128
        #print(model.summary())

        #im_name = 'data/perfectly_detected_ears/test/0001.png'
        #img = cv2.imread(im_name)
        #img_array = cv2.resize(img, (128,128))
        #img_array = np.expand_dims(img_array, axis=0)
        #prediction = model.predict(img_array)
        #print(prediction)
        #print('Predicted class:', np.argmax(prediction) + 1)
        #print('Actual class:', cla_d['/'.join(im_name.split('/')[-2:])])

        # Pixel-wise comparison:
        import feature_extractors.pix2pix.extractor as p2p_ext
        pix2pix = p2p_ext.Pix2Pix()

        import feature_extractors.lbp.extractor as lbp
        lbp = lbp.LBP()
        
        lbp_features_arr = []
        plain_features_arr = []
        y = []
        predictions_arr = []

        for im_name in im_list:
            
            # Read an image
            img = cv2.imread(im_name)

            y.append(cla_d['/'.join(im_name.split('/')[-2:])])

            # Apply some preprocessing here
            #img = preprocess.histogram_equlization_rgb(img)
            #img = preprocess.sharpen(img)

            img_array = cv2.resize(img, (img_size,img_size))
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            #print(prediction.shape)
            predictions_arr.append(prediction)

            # Run the feature extractors            
            #plain_features = pix2pix.extract(img)
            #plain_features_arr.append(plain_features)
            
            #lbp_features = lbp.extract(img)
            #lbp_features_arr.append(lbp_features)

        r0 = eval.my_compute_rank1(predictions_arr, y)
        print('Accuracy Rank-1[%]', r0)

        r5 = eval.my_compute_rank5(predictions_arr, y)
        print('Accuracy Rank-5[%]', r5)

        rankN_arr = []
        for n in range(100):
            rN = eval.my_compute_rankN(predictions_arr, y, n+1)
            #print('Accuracy Rank-N[%]', rN)
            rankN_arr.append(rN)

        # for plotting CMC curve
        resnet50Array = [11.200000000000001, 16.400000000000002, 21.2, 25.2, 28.4, 31.6, 32.0, 34.4, 37.2, 40.400000000000006, 40.8, 42.0, 43.2, 44.800000000000004, 46.800000000000004, 47.599999999999994, 48.4, 48.8, 50.0, 51.6, 52.800000000000004, 53.6, 54.800000000000004, 57.199999999999996, 61.6, 63.2, 63.2, 63.2, 64.0, 65.2, 65.2, 66.4, 68.0, 70.0, 72.0, 72.39999999999999, 74.0, 74.8, 75.6, 76.8, 77.2, 77.2, 77.60000000000001, 78.4, 80.0, 80.80000000000001, 81.2, 82.0, 82.0, 82.8, 82.8, 82.8, 83.2, 83.6, 84.8, 85.2, 85.6, 85.6, 86.4, 87.2, 87.2, 88.8, 89.60000000000001, 89.60000000000001, 89.60000000000001, 89.60000000000001, 89.60000000000001, 90.0, 90.8, 90.8, 91.2, 91.60000000000001, 92.4, 92.4, 92.4, 92.80000000000001, 92.80000000000001, 93.60000000000001, 94.0, 94.8, 95.19999999999999, 96.0, 96.0, 96.0, 96.39999999999999, 96.39999999999999, 96.8, 97.6, 98.4, 98.4, 98.8, 98.8, 98.8, 98.8, 99.2, 99.2, 99.2, 99.6, 100.0, 100.0]
        vgg16Array = [6.0, 8.799999999999999, 12.0, 15.6, 17.599999999999998, 20.8, 24.4, 26.8, 28.799999999999997, 30.4, 31.6, 33.2, 34.4, 36.0, 36.4, 37.2, 38.0, 39.6, 40.0, 42.4, 43.6, 44.800000000000004, 45.2, 46.400000000000006, 47.599999999999994, 48.0, 49.6, 51.6, 52.0, 53.2, 54.0, 54.800000000000004, 55.2, 55.60000000000001, 56.00000000000001, 56.39999999999999, 59.199999999999996, 60.4, 61.199999999999996, 61.6, 62.0, 62.0, 62.8, 64.4, 65.2, 65.60000000000001, 66.0, 67.60000000000001, 69.19999999999999, 70.8, 71.6, 72.39999999999999, 73.6, 74.8, 74.8, 74.8, 75.2, 76.4, 77.60000000000001, 78.4, 78.8, 79.2, 79.2, 80.0, 80.4, 80.80000000000001, 81.2, 81.6, 81.6, 81.6, 82.39999999999999, 82.8, 83.6, 85.2, 85.6, 85.6, 86.0, 86.8, 87.2, 87.6, 89.2, 90.8, 92.4, 93.2, 94.0, 94.39999999999999, 94.8, 94.8, 95.19999999999999, 95.6, 96.0, 96.0, 96.8, 97.2, 97.2, 98.4, 98.4, 99.2, 99.6, 100.0]
        resnet101Array = [7.199999999999999, 13.200000000000001, 17.2, 21.6, 24.0, 25.2, 29.599999999999998, 31.6, 34.4, 36.8, 40.0, 42.0, 42.8, 44.0, 45.2, 47.199999999999996, 48.8, 50.0, 51.6, 53.6, 55.2, 56.39999999999999, 57.99999999999999, 58.4, 59.599999999999994, 60.0, 60.0, 60.8, 62.0, 63.2, 63.2, 64.0, 65.60000000000001, 66.4, 67.2, 68.4, 70.8, 71.2, 72.0, 73.2, 73.6, 75.6, 76.4, 78.0, 78.8, 79.60000000000001, 80.0, 80.80000000000001, 81.2, 81.6, 83.2, 83.6, 84.0, 84.8, 85.2, 85.6, 86.0, 86.0, 87.2, 87.6, 88.0, 88.4, 88.8, 89.60000000000001, 89.60000000000001, 89.60000000000001, 89.60000000000001, 90.0, 90.0, 90.8, 91.60000000000001, 91.60000000000001, 92.0, 92.0, 92.0, 92.4, 92.80000000000001, 93.60000000000001, 94.39999999999999, 94.8, 95.19999999999999, 95.19999999999999, 95.19999999999999, 95.19999999999999, 95.19999999999999, 95.19999999999999, 96.0, 96.39999999999999, 96.39999999999999, 97.6, 97.6, 97.6, 98.0, 98.4, 98.8, 99.2, 99.6, 100.0, 100.0, 100.0]
        resnet152Array = [11.600000000000001, 18.0, 23.200000000000003, 28.4, 33.2, 34.8, 38.0, 40.0, 41.199999999999996, 44.4, 47.199999999999996, 48.4, 50.4, 53.6, 54.800000000000004, 56.8, 57.99999999999999, 59.199999999999996, 60.4, 62.4, 63.2, 63.6, 64.0, 64.8, 65.2, 65.60000000000001, 66.0, 68.0, 68.8, 70.8, 71.6, 72.8, 74.4, 75.6, 76.4, 76.8, 77.2, 77.60000000000001, 78.4, 78.8, 79.2, 80.0, 80.80000000000001, 82.0, 82.0, 82.39999999999999, 82.8, 83.6, 84.39999999999999, 84.39999999999999, 85.2, 86.0, 86.4, 86.4, 86.4, 87.2, 87.2, 88.4, 88.8, 88.8, 89.60000000000001, 90.0, 91.60000000000001, 92.4, 92.80000000000001, 92.80000000000001, 93.2, 93.60000000000001, 93.60000000000001, 94.39999999999999, 94.8, 95.19999999999999, 95.6, 96.0, 96.0, 96.0, 96.0, 96.39999999999999, 96.39999999999999, 96.39999999999999, 96.39999999999999, 96.39999999999999, 97.2, 97.6, 97.6, 97.6, 97.6, 97.6, 97.6, 98.0, 98.4, 98.8, 98.8, 99.2, 99.2, 99.6, 100.0, 100.0, 100.0, 100.0]

        fig = plt.figure()
        #plt.plot(rankN_arr, label="resnet50")
        plt.plot(resnet50Array, label = "ResNet50")
        plt.plot(resnet101Array, label = "ResNet101")
        plt.plot(resnet152Array, label = "ResNet152")
        plt.plot(vgg16Array, label = "VGG16")
        plt.xlabel("RankN")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        
        #Y_plain = cdist(plain_features_arr, plain_features_arr, 'jensenshannon')
        
        #r1 = eval.compute_rank1(Y_plain, y)
        #print('Pix2Pix Rank-1[%]', r1)

        #Z_plain = cdist(lbp_features_arr, lbp_features_arr, 'jensenshannon')
        
        #r2 = eval.compute_rank1(Z_plain, y)
        #print('LBP Rank-1[%]', r2)

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()