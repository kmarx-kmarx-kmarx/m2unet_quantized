import numpy as np
import cv2
import glob

labels_str = "octopi-labels_"
pred_str = "octopi-pred-labels_"
imgs = glob.glob("*.png")
n_labels = len([i for i in imgs if labels_str in i])

def jaccard_sim(img1, img2):
    n = np.prod(img1.shape)
    a = img1 * img2
    b = img1 + img2 - a
    J = a/b
    J[np.isnan(J)] = 1
    j = np.sum(J)/n

    return j


j = []
for i in range(n_labels):
    label = labels_str + str(i) + ".png"
    pred  = pred_str + str(i) + ".png"

    i_pred = np.array(cv2.imread(pred)[:,:,0], dtype='f')
    i_label = np.array(cv2.imread(label)[:,:,0], dtype='f')

    i_pred = i_pred/255.0
    i_label = i_label/255.0

    j.append(jaccard_sim(i_label, i_pred))
    
print(j)
print(min(j))
print(max(j))