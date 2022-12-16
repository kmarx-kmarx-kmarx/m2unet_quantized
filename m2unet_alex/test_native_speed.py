import os
from interactive_m2unet import M2UnetInteractiveModel

import numpy as np
import albumentations as A
import time
from skimage.filters import threshold_otsu
import torch

print(f'GPU: {torch.cuda.is_available()}')

model_root = "./models"
diam_mean = 30

epochs = 1
steps = 1
resume = True
corrid = "200"

transform = A.Compose(
    [
        A.Rotate(limit=180, p=1),
        A.HorizontalFlip(p=1),
    ]
)
A.save(transform, "./transform.json")
pretrained_model = None  # os.path.join(model_root, str(corrid), "model.h5")
os.makedirs(os.path.join(model_root, str(corrid)), exist_ok=True)
# unet model hyperparamer can be found here: https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=f899f7a8a9144b3f946c4a1362f7e38ae0c00c59&device=unknown_device&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f79696e676b61697368612f6b657261732d756e65742d636f6c6c656374696f6e2f663839396637613861393134346233663934366334613133363266376533386165306330306335392f6578616d706c65732f757365725f67756964655f6d6f64656c732e6970796e62&logged_in=true&nwo=yingkaisha%2Fkeras-unet-collection&path=examples%2Fuser_guide_models.ipynb&platform=mac&repository_id=323426984&repository_type=Repository&version=95#Swin-UNET
model_config = {
    "type": "m2unet",
    "activation": "sigmoid",
    "output_channels": 1,
    "loss": {"name": "BCELoss", "kwargs": {}},
    "optimizer": {"name": "RMSprop", "kwargs": {"lr": 1e-2, "weight_decay": 1e-8, "momentum": 0.9}},
    "augmentation": A.to_dict(transform),
}
model = M2UnetInteractiveModel(
    model_config=model_config,
    model_dir=model_root,
    resume=resume,
    pretrained_model=pretrained_model,
    default_save_path=os.path.join(model_root, str(corrid), "model.pth"),
)

start = time.time()
inputs = np.zeros([1, 512, 512, 3], dtype='float32')

rounds = 10
for j in range(rounds):
    results = model.predict(inputs)
    mask_img = results[0, :, :, :]
    threshold = threshold_otsu(mask_img)
    mask_img[mask_img <= threshold] = 0

print(f'time: {(time.time() - start)/rounds}, image shape: {inputs.shape}')