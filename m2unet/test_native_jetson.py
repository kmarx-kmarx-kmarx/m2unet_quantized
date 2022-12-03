import os
from interactive_m2unet import M2UnetInteractiveModel

import numpy as np
import imageio
import albumentations as A
from skimage.filters import threshold_otsu
from skimage.measure import label
import torch
import time

# check if GPU is available
print(f'GPU: {torch.cuda.is_available()}')

t0 = time.time()

def jaccard_sim(img1, img2):
    n = np.prod(img1.shape)
    a = img1 * img2
    b = img1 + img2 - a
    J = a/b
    J[np.isnan(J)] = 1
    j = np.sum(J)/n

    return j

# setting up
data_dir = '../../test/072622-D8-9_2022-07-27_18-51-3.318936' # data should contain a train and a test folder
model_root = "../models_100"
epochs = 100
steps = 1
resume = True
corrid = "200"
pretrained_model = None  # os.path.join(model_root, str(corrid), "model.h5")
os.makedirs(os.path.join(model_root, str(corrid)), exist_ok=True)
sz = 1024

model = M2UnetInteractiveModel(
    model_dir=model_root,
    resume=resume,
    pretrained_model=pretrained_model,
    default_save_path=os.path.join(model_root, str(corrid), "model.pth"),
)

npy_files = [os.path.join(data_dir + '/0', s) for s in os.listdir(data_dir + '/0') if s.endswith('.npy')]
save = "./results.csv"
# test
with open(save, 'w') as f:
    for i, file in enumerate(npy_files):
        t1 = time.time()
        
        # Load file
        try:
            items = np.load(file, allow_pickle=True).item()
        except:
            print("Bad Item")
            continue
        mask = (items['masks'][:, :, None]  > 0) * 1.0
        outline = (items['outlines'][:, :, None]  > 0) * 1.0
        mask = mask * (1.0 - outline)
        sample = (items['img'], mask)

        inputs = sample[0].astype("float32")[None, :sz, :sz, :]
        labels = sample[1].astype("float32")[None, :sz, :sz, :] * 255
        results = model.predict(inputs)
        output = np.clip(results[0] * 255, 0, 255)[:, :, 0].astype('uint8')
        threshold = threshold_otsu(output)
        mask_new = ((output > threshold) * 255).astype('uint8')
        predict_labels = label(mask_new)
        t2 = time.time()
        dt = t2-t1
        imageio.imwrite(f"octopi-pred-labels_{i}.png", predict_labels)
        imageio.imwrite(f"octopi-pred-prob_{i}.png", output)
        imageio.imwrite(f"octopi-labels_{i}.png", labels[0].astype('uint8'))
        a = np.max(label(mask))
        b = np.max(predict_labels)
        j = jaccard_sim(results[0], sample[1].astype("float32")[None, :sz, :sz, :])
        f.write(f'{dt},{a},{b},{j}\n')

print("all done")
