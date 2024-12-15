from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import h5py
from config import Config

class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, tokenizer, size=Config.image_size):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.size = size

        self.h5_path = Config.data_train_path
        with h5py.File(self.h5_path, "r") as f:
            self.captions = f["train_cap"][:]
            self.img_ids = f["train_imid"][:]
            word_code = f["word_code"][0]

        # Build vocabulary mappings
        self.idx_to_word = {}
        for word in word_code.dtype.fields:
            idx = word_code[word][()]
            self.idx_to_word[idx] = word

        # Filter out invalid images
        valid_indices = self.filter_valid_images()
        self.captions = self.captions[valid_indices]
        self.img_ids = self.img_ids[valid_indices]

        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # Convert caption indices to words
        caption_indices = self.captions[idx]
        words = [self.idx_to_word[int(idx)] for idx in caption_indices if int(idx) in self.idx_to_word]
        caption_text = ' '.join(words)

        # Load image
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.root_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Tokenize caption
        inputs = self.tokenizer(
            caption_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
        }

    def filter_valid_images(self):
        valid_indices = []
        for i, img_id in enumerate(self.img_ids):
            img_path = os.path.join(self.root_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                valid_indices.append(i)
        return valid_indices