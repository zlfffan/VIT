from typing import Optional, Callable, Tuple
from torch.utils.data import Dataset
import numpy as np
import pickle
import os


class Mydataset(Dataset):
	def __init__(self,
	             data_root: str = "./CIFAR10",
	             train: bool = True,
	             transform: Optional[Callable] = None,
	             target_transform: Optional[Callable] = None) -> None:
		self.data_root = data_root
		self.obj_img_label(train=train)
		self.obj_classes_to_idx()
		self.transform = transform
		self.target_transform = target_transform
	
	def __getitem__(self, idx: int) -> Tuple[np.array, int]:
		img = self.imgs[idx]
		label = self.labels[idx]
		if self.transform:
			img = self.transform(img)
		if self.target_transform:
			label = self.target_transform(label)
		return img, label
	
	def __len__(self) -> int:
		return len(self.labels)
	
	def obj_classes_to_idx(self) -> None:
		with open(os.path.join(self.data_root, "batches.meta"), "rb") as f:
			img_dict = pickle.load(f, encoding="bytes")
			self.classes_to_idx = {img_dict[b'label_names'][i].decode(): i for i in range(len(img_dict[b'label_names']))}
	
	def obj_img_label(self, train) -> None:
		self.labels = []
		self.imgs = np.empty((0, 3 * 32 * 32), dtype=np.uint8)
		if train:
			for dirs in os.listdir(self.data_root):
				if dirs.startswith("data_batch_"):
					with open(os.path.join(self.data_root, dirs), "rb") as f:
						img_dict = pickle.load(f, encoding="bytes")
						self.labels += img_dict[b'labels']
						self.imgs = np.concatenate((self.imgs, img_dict[b'data']), axis=0)
		else:
			with open(os.path.join(self.data_root, "test_batch"), "rb") as f:
				img_dict = pickle.load(f, encoding="bytes")
				self.labels += img_dict[b'labels']
				self.imgs = np.concatenate((self.imgs, img_dict[b'data']), axis=0)
		self.imgs = self.imgs.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
