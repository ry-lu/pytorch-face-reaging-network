
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from itertools import permutations

TARGET_AGES = [18,23,28,33,38,43,48,53,58,63,68,73,78,83]

def generate_image_pairs(image_meta):
    """List of all possible age transitions for a person."""
    image_pairs = []
    for _, row in image_meta.iterrows():
        image_paths = row.tolist()
        image_combinations = permutations(image_paths, 2)
        image_pairs.extend(list(image_combinations))
    return image_pairs

def generate_age_image_dict(image_meta):
    """
    Returns:
        image_age_dict: {image path:age}
    """
    image_age_dict={}
    for target_age, column in zip(TARGET_AGES,image_meta.columns):
        image_age_dict.update({path:target_age for path in image_meta[column].values})
    return image_age_dict

class FRANDataset(Dataset):
    def __init__(
        self,
        image_meta,
        transforms,
        data_dir,
    ):
        self.image_pairs = generate_image_pairs(image_meta)
        self.image_ages_dict = generate_age_image_dict(image_meta)
        self.transforms = transforms
        self.data_dir = data_dir

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        image_pair = self.image_pairs[index]

        input_image = np.array(Image.open(self.data_dir/image_pair[0]).convert('RGB'))
        target_image = np.array(Image.open(self.data_dir/image_pair[1]).convert('RGB'))

        normalized_input_image_transformed = self.transforms(image=input_image)
        # Basic normalization
        normalized_input_image = normalized_input_image_transformed['image']/127.5 - 1

        # Replay augmentations on second image
        normalized_target_image = A.ReplayCompose.replay(
            normalized_input_image_transformed['replay'], 
            image=np.array(target_image),
            )['image']/127.5 - 1

        # Get age maps of both images
        _,width,height=normalized_input_image.shape
        age_map1 = torch.full((1,width,height),self.image_ages_dict[image_pair[0]]/100)
        age_map2 = torch.full((1,width,height),self.image_ages_dict[image_pair[1]]/100)

        # Combine RGB delta diff with age maps for 5-channel tensor
        input_tensor = torch.cat((normalized_input_image, age_map1, age_map2), dim=0)

        return {
            'input': input_tensor,
            'normalized_input_image': normalized_input_image,
            'normalized_target_image': normalized_target_image,
            'target_age': age_map2,
        }