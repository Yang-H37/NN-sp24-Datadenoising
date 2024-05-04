import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

class Denoising_Autoencoder(nn.Module):
    def __init__(self):
        super(Denoising_Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        # Decoder
        x = self.decoder(x)
        return x

# SSIM loss function
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        from pytorch_msssim import SSIM as _SSIM
        self.ssim = _SSIM(data_range=1, size_average=True, win_size=7, channel=3)

    def forward(self, x, y):
        # Calculate SSIM loss for each channel separately
        ssim_loss = 1 - self.ssim(x, y)
        # Take the mean across the batch
        ssim_loss = torch.mean(ssim_loss)
        return ssim_loss

class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, target_size=(256, 256)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.image_filenames = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        gt_img_name = os.path.join(self.root_dir, self.image_filenames[idx])
        noisy_img_name = gt_img_name.replace("GT_SRGB", "NOISY_SRGB")

        gt_image = Image.open(gt_img_name).convert("RGB")
        noisy_image = Image.open(noisy_img_name).convert("RGB")

        # Resize images to the target size
        if self.target_size:
            gt_image = transforms.Resize(self.target_size)(gt_image)
            noisy_image = transforms.Resize(self.target_size)(noisy_image)

        if self.transform:
            gt_image = self.transform(gt_image)
            noisy_image = self.transform(noisy_image)

        return gt_image, noisy_image

# Load the trained model
model = Denoising_Autoencoder()
model.load_state_dict(torch.load('denoising_autoencoder.pth', map_location=torch.device('cpu')))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load test dataset
test_root_dir = "dataset/long_png_resized"
test_dataset = ImagePairDataset(test_root_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate the model on the test set
device = torch.device("cpu")
criterion = SSIM().to(device)
test_losses = []

output_dir = 'test_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with torch.no_grad():
    for idx, (gt_images, noisy_images) in enumerate(test_loader):
        gt_images = gt_images.to(device)
        noisy_images = noisy_images.to(device)

        outputs = model(noisy_images)
        loss = criterion(outputs, gt_images)
        test_losses.append(loss.item())

        # Save denoised images
        denoised_image = outputs[0].permute(1, 2, 0).cpu().detach().numpy()
        denoised_image = (denoised_image * 255).astype('uint8')
        Image.fromarray(denoised_image).save(os.path.join(output_dir, f'denoised_image_{idx}.png'))


avg_test_loss = sum(test_losses) / len(test_losses)
print(f"Average SSIM Loss of Test Dataset: {avg_test_loss:.4f}")

# Plot the SSIM loss of the test dataset
plt.plot(test_losses)
plt.xlabel('Image')
plt.ylabel('SSIM Loss')
plt.title('SSIM Loss of Test Dataset')
plt.savefig(os.path.join(output_dir, 'ssim_loss_plot.png'))
plt.close()
