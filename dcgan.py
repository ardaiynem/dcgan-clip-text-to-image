import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from torchvision.utils import save_image
import os
from dataset import ImageCaptionDataset
from config import Config
from models.generator import Generator
from models.discriminator import Discriminator


def main():
    # Initialize tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(Config.device)
    text_encoder.eval()  # Set to eval mode if not fine-tuning

    # Initialize dataset and dataloader
    dataset = ImageCaptionDataset(Config.image_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=Config.train_batch_size, shuffle=True, num_workers=Config.num_workers)

    # Initialize models
    netG = Generator(Config.nz, Config.ngf, Config.nc, Config.embedding_dim).to(Config.device)
    netD = Discriminator(Config.nc, Config.ndf, Config.embedding_dim).to(Config.device)

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=Config.learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=Config.learning_rate, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(Config.num_epochs):
        for i, batch in enumerate(tqdm(dataloader)):
            batch_size = batch["pixel_values"].size(0)
            real_labels = torch.ones(batch_size, device=Config.device)
            fake_labels = torch.zeros(batch_size, device=Config.device)

            # Prepare data
            images = batch["pixel_values"].to(Config.device)
            captions = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)

            # Get text embeddings from CLIP (without gradient)
            with torch.no_grad():
                text_outputs = text_encoder(input_ids=captions, attention_mask=attention_mask)
                embeddings = text_outputs.last_hidden_state.mean(dim=1)  # (batch_size, embedding_dim)

            # Train Discriminator with real images
            netD.zero_grad()
            outputs = netD(images, embeddings)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            # Generate fake images
            noise = torch.randn(batch_size, Config.nz, device=Config.device)
            noise_embeddings = torch.randn_like(embeddings)
            fake_images = netG(noise, embeddings.detach())

            # Train Discriminator with fake images
            outputs = netD(fake_images.detach(), embeddings.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            outputs = netD(fake_images, embeddings.detach())
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizerG.step()

        # Save sample images
        if (epoch + 1) % 10 == 0:
            os.makedirs(Config.output_dir, exist_ok=True)
            save_image(fake_images.data[:25], os.path.join(Config.output_dir, f'image_epoch_{epoch + 1}.png'), nrow=5, normalize=True)

# Example usage
if __name__ == '__main__':
    main()