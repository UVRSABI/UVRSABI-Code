import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg

from typing import Any, Callable, Optional, Tuple

EXTENSIONS = ['.jpg', '.png', '.JPG']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])



class VOCSegmentation(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super(VOCSegmentation, self).__init__(root, transforms, transform, target_transform)
        # valid_sets = ["train", "trainval", "val",'dataset']
        # self.image_set = verify_str_arg(image_set, "image_set", valid_sets)
        # voc_root = os.path.join(self.root, self.image_set)

        self.images_root = root

        self.filenames = [image_basename(f)
            for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames.sort()

        # self.filenamesGt = None
        # self.labels_root = None

        # if os.path.isdir(self.labels_root):
        #     self.filenamesGt = [image_basename(f)
        #         for f in os.listdir(self.labels_root) if is_image(f)]
            
        #     print(self.filenamesGt)
        #     self.filenamesGt.sort()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        filename = self.filenames[index]


        img_path = image_path(self.images_root, filename, '.png')
        if not os.path.exists(img_path):
            img_path = image_path(self.images_root, filename, '.jpg')
        if not os.path.exists(img_path):
            img_path = image_path(self.images_root, filename, '.png')
        
        with open(img_path, 'rb') as f:
            image = load_image(f).convert('RGB')


  
        if self.transforms is not None:
            label = None
            image, label = self.transforms(image,label)


        return image, filename

    def ImageSize(self, index: int) -> Tuple[int, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (height, width) of the image.
        """
        filename = self.filenames[index]
        img_path = image_path(self.images_root, filename, '.png')
        if not os.path.exists(img_path):
            img_path = image_path(self.images_root, filename, '.jpg')
        if not os.path.exists(img_path):
            img_path = image_path(self.images_root, filename, '.JPG')
        img = Image.open(img_path).convert('RGB')
        return img.size
    def FileName(self, index: int) -> str:
        """
        Args:
            index (int): Index

        Returns:
            str: filename of the image.
        """
        filename = self.filenames[index]

        img_path = image_path(self.images_root, filename, '.png')
        if not os.path.exists(img_path):
            img_path = image_path(self.images_root, filename, '.jpg')
        if not os.path.exists(img_path):
            img_path = image_path(self.images_root, filename, '.JPG')

        return img_path.split('/')[-1]
    def __len__(self) -> int:
        return len(self.filenames)