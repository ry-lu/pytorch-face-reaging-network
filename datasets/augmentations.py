# Some augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform_normalize = A.ReplayCompose([
    A.Resize(512, 512),
    # Crop is really important
    A.RandomCrop(256, 256),
    A.ColorJitter(p=0.8),
    A.Blur(p=0.1),
    A.RandomBrightnessContrast(),
    A.Affine(rotate=[-30, 30],scale=(0.5,1.5), p=0.8),
    ToTensorV2(),
])
