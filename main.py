import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
        # Calculate SSIM loss
        ssim_loss = 1 - self.ssim(x, y)
        return ssim_loss

class ImagePairDataset(Dataset):
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


# Define training parameters and data loaders
root_dir = "dataset/train/"
val_root_dir = "dataset/val/"
batch_size = 4
num_epochs = 10
lr = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
])

target_size = (256, 256)
dataset = ImagePairDataset(root_dir, transform=transform, target_size=target_size)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

val_dataset = ImagePairDataset(val_root_dir, transform=transform, target_size=target_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Denoising_Autoencoder().to(device)
criterion = SSIM().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (gt_images, noisy_images) in enumerate(train_loader):
        gt_images = gt_images.to(device)
        noisy_images = noisy_images.to(device)

        optimizer.zero_grad()

        outputs = model(noisy_images)
        loss = criterion(outputs, gt_images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}")
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for gt_images, noisy_images in val_loader:
            gt_images = gt_images.to(device)
            noisy_images = noisy_images.to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, gt_images)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss}")
    val_losses.append(avg_val_loss)

# Save the trained model
torch.save(model.state_dict(), 'denoising_autoencoder.pth')

# Plotting training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()