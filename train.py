import os
from torch.utils import tensorboard
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
import Dataset
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, CenterCrop, RandomHorizontalFlip, RandomErasing
import torch.nn as nn
import sys
import torch
import vit_model
import argparse


def train(args):
	num_cls = args.num_cls
	epochs = args.epochs
	batch_size = args.batch_size
	lr_init = args.lr_init
	iteration = 0
	transform = {"train": Compose([ToTensor(),
	                               Resize((224, 224), antialias=True),
	                               Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
	             "val": Compose([ToTensor(),
	                             Resize((224, 224), antialias=True),
	                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	train_data = Dataset.Mydataset(train=True, transform=transform['train'])
	test_data = Dataset.Mydataset(train=False, transform=transform['val'])
	print("using {} images for training, {} images for validation.".format(len(train_data), len(test_data)))
	train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
	test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
	len_batch = len(test_dataloader)
	test_size = len(test_data)
	best_acc = 0
	
	model = vit_model.VisionTransformer(img_size=224,  # size of images
	                                    patch_size=32,  # the size images divided by 32
	                                    embed_dim=768,  # token dim 768
	                                    depth=12,  # Lx 12
	                                    num_heads=12,  # multi_head 12
	                                    representation_size=None,
	                                    num_classes=num_cls,
	                                    drop_ratio=0)  # use the pre_train model of ImageNet
	"""
	预训练模型下载
	链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg
	提取密码: s5hl
	"""
	optimizer = optim.SGD(model.parameters(), lr=0, momentum=0.9)
	
	if args.use_ckpt:
		try:
			assert os.path.exists(args.save_root + args.use_ckpt), print("checkpoint does not exist")
			ckpt = torch.load(args.save_root + args.use_ckpt)
			model.load_state_dict(ckpt["model"])
			optimizer.load_state_dict(ckpt["optimizer"])
			iteration = ckpt["iteration"] + 1
			print("init model by checkpoint")
		except:
			pass
	if args.pretrain:
		model.head = nn.Linear(model.num_features, 1000)
		model.load_state_dict(torch.load(args.save_root + args.pretrain))
		model.head = nn.Linear(model.num_features, num_cls)
		print("init model by pretrained model")
	else:
		print("init model randomly")
	model = model.to(device)
	
	loss_func = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0, momentum=0.9)
	
	writer = tensorboard.SummaryWriter(args.save_root + "VIT")
	# tensorboard --logdir args.save_root + "VIT"
	
	for epoch in range(iteration, epochs):
		lr = model.update_lr(optimizer, lr_init, epoch, epochs)
		model.train()
		train_bar = tqdm(train_dataloader, file=sys.stdout, unit=" batches")
		train_correct = 0
		train_loss = 0
		train_acc = 0
		for batch, (imgs, labels) in enumerate(train_bar):
			imgs = imgs.to(device)
			labels = labels.to(device)
			predict = model(imgs)
			train_correct += (predict.argmax(1) == labels).sum().item()
			train_acc = train_correct / ((batch + 1) * batch_size)
			loss = loss_func(predict, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss = (batch * train_loss + loss.item()) / (batch + 1)
			train_bar.desc = "train epoch[{}/{}] train_acc:{:.3f} train_loss:{:.3f} lr:{:.6f}".format(epoch + 1, epochs, train_acc, train_loss, lr)
		
		model.eval()
		val_acc = 0
		correct = 0
		val_loss = 0
		with torch.no_grad():
			val_bar = tqdm(test_dataloader, file=sys.stdout, unit=" batches")
			for imgs, labels in val_bar:
				imgs = imgs.to(device)
				labels = labels.to(device)
				predict = model(imgs)
				correct += (predict.argmax(1) == labels).sum().item()
				val_loss += loss_func(predict, labels).item()
			val_loss /= len_batch
			val_acc = correct / test_size
		print('[test epoch %d]  val_acc: %.3f  val_loss: %.3f  ' % (epoch + 1, val_acc, val_loss))
		
		# --------------save the best model--------------------
		if val_acc > best_acc:
			best_acc = val_acc
			torch.save(model.state_dict(), args.save_root + args.model_name)
		
		if args.save_ckpt:
			torch.save({"model": model.state_dict(), "iteration": epoch, "optimizer": optimizer.state_dict()}, args.save_root + args.save_ckpt)
		
		writer.add_scalars('loss', {"train": round(train_loss, 3), "val": round(val_loss, 3)}, epoch + 1)
		writer.add_scalars('acc', {"train": round(train_acc, 3), "val": round(val_acc, 3)}, epoch + 1)
	
	writer.close()
	print('Finished Training')


if __name__ == "__main__":
	parser = argparse.ArgumentParser("Vision Transformer Training")
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--num_cls', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--lr_init', type=float, default=5e-3)
	parser.add_argument('--model_name', type=str, default='VIT.pth', help='model name')
	parser.add_argument('--save_root', type=str, default='./model_pre/', help='path to save model')
	parser.add_argument('--pretrain', type=str, default="pretrained_model.pth", help='path to load pretrained model')
	parser.add_argument('--save_ckpt', type=str, default=None, help='path to save checkpoint')
	parser.add_argument('--use_ckpt', type=str, default=None, help='path to load checkpoint')
	args = parser.parse_args()
	
	train(args)
