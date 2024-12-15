import torch

class Config:
    data_train_path = "data/eee443_project_dataset_train.h5"  # Provided dataset
    output_dir = "./logs"
    image_dir = "data/images"  # Directory to store images
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    image_size = 64
    train_batch_size = 64
    num_epochs = 50
    learning_rate = 0.0002
    nz = 100 # Size of latent vector
    ngf = 64 # Size of feature maps in generator
    ndf = 64 # Size of feature maps in discriminator
    nc = 3 # Number of channels in the training images
    embedding_dim = 768  # CLIP's embedding dimension for 'openai/clip-vit-large-patch14'
    num_workers = 4