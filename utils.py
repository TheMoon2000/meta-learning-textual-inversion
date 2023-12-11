import face_recognition
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image
from torch import nn
import cv2
import os, random
from torchvision import transforms
from typing import Union
import tqdm
import torch.nn.init as init

embedding_cache = {}

def get_face_embedding(image: Union[str, np.ndarray]):
    if isinstance(image, str):
        if image in embedding_cache:
            return embedding_cache[image]
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image)
    # face_landmarks = face_recognition.face_landmarks(image, face_locations)
    
    # Check if a face is detected
    if not face_locations:
        # print("No face found in the provided image.")
        return 0, np.zeros(128)
    
    face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=1)

    # Check if a face encoding is found
    if not face_encodings:
        # print("Face encoding not found.")
        return 0, np.zeros(128)
    
    # Take the first face encoding (assuming there's only one face in the image)
    face_embedding = face_encodings[0]

    # Convert the face embedding to a fixed-length numpy array
    fixed_length_embedding = np.array(face_embedding)

    if isinstance(image, str):
        embedding_cache[image] = fixed_length_embedding
    
    return 1, fixed_length_embedding

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

class TextualInversionDataset(Dataset):

    imagenet_templates_small = [
        # "a photo of a {}",
        # "a rendering of a {}",
        # "a cropped photo of the {}",
        # "the photo of a {}",
        # "a photo of a clean {}",
        # "a photo of a dirty {}",
        # "a dark photo of the {}",
        # "a photo of the cool {}",
        # "a close-up photo of a {}",
        # "a bright photo of the {}",
        # "a cropped photo of a {}",
        # "a photo of the {}",
        # "a good photo of the {}",
        # "a photo of one {}",
        # "a close-up photo of the {}",
        # "a rendition of the {}",
        # "a photo of the clean {}",
        # "a rendition of a {}",
        # "a photo of a nice {}",
        # "a good photo of a {}",
        # "a photo of the nice {}",
        # "a photo of the small {}",
        # "a photo of the weird {}",
        # "a photo of the large {}",
        # "a photo of a cool {}",
        "a photo of {}",
        "a picture of {}",
        "{}",
        "a close-up photo of {}",
        "a cropped photo of {}"
        # "a photo of a small {}",
    ]

    imagenet_style_templates_small = [
        "a painting in the style of {}",
        "a rendering in the style of {}",
        "a cropped painting in the style of {}",
        "the painting in the style of {}",
        "a clean painting in the style of {}",
        "a dirty painting in the style of {}",
        "a dark painting in the style of {}",
        "a picture in the style of {}",
        "a cool painting in the style of {}",
        "a close-up painting in the style of {}",
        "a bright painting in the style of {}",
        "a cropped painting in the style of {}",
        "a good painting in the style of {}",
        "a close-up painting in the style of {}",
        "a rendition in the style of {}",
        "a nice painting in the style of {}",
        "a small painting in the style of {}",
        "a weird painting in the style of {}",
        "a large painting in the style of {}",
    ]
    
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        skip_faceid=False
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.placeholder_token = placeholder_token
        self.flip_p = flip_p
        self.skip_ids = skip_faceid
        self.images = []
        self.identities = []
        self.images_by_identity = [[] for _ in range(100)] # range(10177)]
        self.centroids = {}
        with open(os.path.join(self.data_root, 'identity_CelebA.txt'), 'r') as f:
            for line in tqdm.tqdm(f.readlines()):
                filename, label = line.strip().split()
                if int(label) > 100: continue
                self.images.append(filename)
                self.identities.append(int(label))
                self.images_by_identity[int(label) - 1].append(filename)
        
        self.interpolation = {
            "linear": Image.AFFINE,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]

        self.templates = self.imagenet_style_templates_small if learnable_property == "style" else self.imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        example = {}
        image = cv2.cvtColor(cv2.imread(os.path.join(self.data_root, 'img_align_celeba', self.images[i])), cv2.COLOR_BGR2RGB)

        # Add face encoding (TODO: test performance before and after augmentation)
        if not self.skip_ids:
            # self.images_by_identity
            if self.identities[i] in self.centroids:
                example["valid"] = True
                if example["valid"]:
                    example["face_id"] = self.centroids[self.identities[i]]
            else:
                path = random.choice(self.images_by_identity[self.identities[i] - 1])
                path = os.path.join(self.data_root, 'img_align_celeba', path)
                example["valid"], example["face_id"] = get_face_embedding(path)

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["prompt"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.skip_ids:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        # image = image.resize((self.size, self.size), resample=self.interpolation)

        # image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        return example


class MetaTextInversion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(128, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 768)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Xavier initialization for linear layers
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
    
    def forward(self, x):
        return self.mlp(x)