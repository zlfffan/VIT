import argparse
import os
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
import vit_model
import torch
import matplotlib.pyplot as plt
import Dataset


def predict(args):
	assert os.path.exists(args.model), print("model is not founded")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	num_cls = args.num_cls
	transform = {"val": Compose([ToTensor(),
	                             Resize((224, 224), antialias=True),
	                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
	test_data = Dataset.Mydataset(train=False, transform=transform['val'])
	test_dataloader = iter(DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0))
	
	model = vit_model.VisionTransformer(img_size=224,  # size of images
	                                    patch_size=32,  # the size images divided by
	                                    embed_dim=768,  # token dim
	                                    depth=12,  # Lx
	                                    num_heads=12,  # multi_head
	                                    representation_size=None,
	                                    num_classes=num_cls)
	model.load_state_dict(torch.load(args.model))
	model = model.to(device)
	idx_to_cls = [cls for cls in test_data.classes_to_idx]
	
	model.eval()
	with torch.no_grad():
		for i in range(69, 90):
			img, label = next(test_dataloader)
			img = img.to(device)
			pre = model(img).softmax(dim=-1)
			plt.imshow(img[0].permute(1, 2, 0).cpu().numpy())
			plt.title(f"Predicted: {idx_to_cls[pre.argmax(dim=-1).item()]}, {pre.max().item():.3f} truth: {idx_to_cls[label.item()]}")
			plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser("Vision Transformer Training")
	parser.add_argument('--model', type=str, default='model_pre/VIT.pth', help='model')
	parser.add_argument('--num_cls', type=int, default=10, help='class number')
	args = parser.parse_args()
	
	predict(args)
