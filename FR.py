import os
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

class FaceRecognition:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))

        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                          thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                          device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.samples_dir = "./rotated"
        self.embeddings_dir = "./embeddings"
        os.makedirs(self.embeddings_dir, exist_ok=True)

        self.embeddings = {}

    def prepare_embeddings(self):
        for filename in os.listdir(self.samples_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(self.samples_dir, filename)
                img = Image.open(img_path)

                img_cropped = self.mtcnn(img)
                
                if img_cropped is not None:
                    img_cropped = img_cropped.to(self.device)
                    img_embedding = self.resnet(img_cropped.unsqueeze(0))
                    embedding_np = img_embedding.detach().cpu().numpy()

                    save_path = os.path.join(self.embeddings_dir, f"{filename.split('.')[0]}_embedding.npy")
                    np.save(save_path, embedding_np)
                    print(f"Embedding for {filename} saved successfully.")
                    self.embeddings[filename.split('_')[0]] = embedding_np
                else:
                    print(f"No face detected in {filename}.")

    def save_embeddings(self, embeddings_filename):
        with open(embeddings_filename, 'wb') as file:
            pickle.dump(self.embeddings, file)

    def load_embeddings(self, embeddings_filename):
        with open(embeddings_filename, 'rb') as file:
            self.embeddings = pickle.load(file)

    def calculate_similarity(self, embedding1):
        similarities = {}
        for name, embedding in self.embeddings.items():
            similarity_cosine = cosine_similarity(embedding1, embedding)
            similarity_euclidean = distance.euclidean(embedding1.flatten(), embedding.flatten())
            similarity_manhattan = distance.cityblock(embedding1.flatten(), embedding.flatten())
            similarities[name] = (similarity_cosine[0][0], similarity_euclidean, similarity_manhattan)
    
        best_candidate = None
        best_cosine_similarity = -1
        best_euclidean_similarity = float('inf')
        best_manhattan_similarity = float('inf')

        for name, (similarity_cosine, similarity_euclidean, similarity_manhattan) in similarities.items():
            if similarity_cosine > best_cosine_similarity or \
                    (similarity_cosine == best_cosine_similarity and
                    similarity_euclidean < best_euclidean_similarity) or \
                    (similarity_cosine == best_cosine_similarity and
                    similarity_euclidean == best_euclidean_similarity and
                    similarity_manhattan < best_manhattan_similarity):
                best_candidate = name
                best_cosine_similarity = similarity_cosine
                best_euclidean_similarity = similarity_euclidean
                best_manhattan_similarity = similarity_manhattan

        return best_candidate, best_cosine_similarity, best_euclidean_similarity, best_manhattan_similarity