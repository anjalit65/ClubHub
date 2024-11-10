import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel
from sklearn.feature_extraction import FeatureHasher
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn

# Load and preprocess images
import os

# Get all image paths from the test_images folder
test_image_folder = "test_images"  # Path to your test_images folder
image_paths = [] 

# Add images from the test_images folder
for file_name in os.listdir(test_image_folder):
    if file_name.endswith((".jpeg", ".jpg", ".png")):  # You can adjust the image formats here
        image_paths.append(os.path.join(test_image_folder, file_name))

print("Updated image paths:", image_paths)
query_image_path = "dog2.jpeg"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Function to normalize features
def normalize_features(features):
    return features / np.linalg.norm(features)

### METHOD 1: Pre-trained CNN (ResNet) + Nearest Neighbor Search ###
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet.eval()

def extract_resnet_features(image_path):
    image = load_image(image_path)
    with torch.no_grad():
        features = resnet(image).squeeze().numpy()
    return normalize_features(features)

# Collect features for ResNet
resnet_features = np.array([extract_resnet_features(img) for img in image_paths])

# Using FAISS for similarity search
resnet_index = faiss.IndexFlatL2(resnet_features.shape[1])
resnet_index.add(resnet_features)

### METHOD 2: Deep Metric Learning (Siamese Network) ###
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        self.fc = nn.Linear(512, 128)
    
    def forward(self, x):
        x = self.feature_extractor(x).view(x.size(0), -1)
        return self.fc(x)

siamese_model = SiameseNetwork()
siamese_model.eval()

def extract_siamese_features(image_path):
    image = load_image(image_path)
    with torch.no_grad():
        features = siamese_model(image).numpy().squeeze()
    return normalize_features(features)

# Collect features for Siamese
siamese_features = np.array([extract_siamese_features(img) for img in image_paths])
siamese_index = faiss.IndexFlatL2(siamese_features.shape[1])
siamese_index.add(siamese_features)

### METHOD 3: Vision Transformer (CLIP) ###
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_clip_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs).numpy().squeeze()
    return normalize_features(features)

# Collect features for CLIP
clip_features = np.array([extract_clip_features(img) for img in image_paths])
clip_index = faiss.IndexFlatL2(clip_features.shape[1])
clip_index.add(clip_features)

### METHOD 4: Hashing (LSH) ###
hasher = FeatureHasher(n_features=256, input_type="string")

def extract_hashed_features(image_path):
    features = extract_resnet_features(image_path)  # Using ResNet features for hashing
    # Convert features into a list of strings to fit the hashing method
    features_str = [str(f) for f in features]
    hash_code = hasher.transform([features_str]).toarray().squeeze()
    return hash_code


hashed_features = np.array([extract_hashed_features(img) for img in image_paths])
hashed_neighbors = NearestNeighbors(n_neighbors=5, metric="cosine").fit(hashed_features)

### METHOD 5: Autoencoder ###
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(224*224*3, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 224*224*3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return x

autoencoder = Autoencoder()
autoencoder.eval()

def extract_autoencoder_features(image_path):
    image = Image.open(image_path).resize((224, 224)).convert("RGB")
    image = np.array(image).flatten() / 255.0
    image_tensor = torch.FloatTensor(image).unsqueeze(0)
    with torch.no_grad():
        features = autoencoder(image_tensor).numpy().squeeze()
    return normalize_features(features)

autoencoder_features = np.array([extract_autoencoder_features(img) for img in image_paths])
autoencoder_index = faiss.IndexFlatL2(autoencoder_features.shape[1])
autoencoder_index.add(autoencoder_features)

### SIMILARITY SEARCH AND COMPARISON ###
def search_and_print_results(query_image_path, method_name, index, features=None):
    if method_name == "LSH":
        query_features = extract_hashed_features(query_image_path).reshape(1, -1)
        D, I = hashed_neighbors.kneighbors(query_features)
    else:
        extract_func = globals()[f"extract_{method_name.lower()}_features"]
        query_features = extract_func(query_image_path).reshape(1, -1)
        D, I = index.search(query_features, k=5)
    print(f"\nTop matches using {method_name}:")
    for idx, (d, i) in enumerate(zip(D[0], I[0])):
        print(f"{idx+1}: Image={image_paths[i]}, Distance={d:.4f}")

# Perform search with each method and display results
print("\n--- Similarity Search Results ---")
search_and_print_results(query_image_path, "resnet", resnet_index)
search_and_print_results(query_image_path, "siamese", siamese_index)
search_and_print_results(query_image_path, "clip", clip_index)
search_and_print_results(query_image_path, "LSH", hashed_neighbors)
search_and_print_results(query_image_path, "autoencoder", autoencoder_index)
