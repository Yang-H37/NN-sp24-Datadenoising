import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import Denoising_Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Denoising_Autoencoder().to(device)
model.load_state_dict(torch.load('denoising_autoencoder.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

val_sample_path = "dataset/single_val/GT_SRGB_001.png"
val_sample = Image.open(val_sample_path).convert("RGB")
val_sample_tensor = transform(val_sample).unsqueeze(0).to(device)

# Denoise the image
with torch.no_grad():
    denoised_image = model(val_sample_tensor).cpu().squeeze(0)

# Plot original, noisy, and denoised images
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(val_sample)
plt.axis('off')

'''
# If have the noisy picture
plt.subplot(1, 3, 2)
plt.title('Noisy')
noisy_image_path = val_sample_path.replace("GT_SRGB", "NOISY_SRGB")
noisy_image = Image.open(noisy_image_path).convert("RGB")
plt.imshow(noisy_image)
plt.axis('off')
'''

plt.subplot(1, 3, 3)
plt.title('Denoised')
denoised_image = transforms.ToPILImage()(denoised_image)
plt.imshow(denoised_image)
plt.axis('off')

plt.show()
