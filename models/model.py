import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image

from .mtcnn import MTCNN
from .inception_resnet_v1 import InceptionResnetV1

def collate_fn(x):
    return x[0]

class FaceNet():
    def __init__(self,model=None):
        self.mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)
        self.mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
        self.name_list = []
        self.embedding_list = []
        
        if model is not None:
            self.__load__(model)
           
    def __load__(self, model_path):
        loaded_data = torch.load(model_path)
        self.embedding_list = loaded_data[0]
        self.name_list = loaded_data[1]       
    
    def train(self, folder_path):
        # Load image from folder
        dataset = datasets.ImageFolder(folder_path)
        idx_to_cls = {i:c for c,i in dataset.class_to_idx.items()}
            
        # Dataloader
        dataloader = DataLoader(dataset, collate_fn=collate_fn)
        
        for img, idx in dataloader:
            face, prob = self.mtcnn0(img, return_prob=True)
        
            if face is not None and prob>0.92:
                emb = self.resnet(face.unsqueeze(0))
                self.embedding_list.append(emb.detach())
                self.name_list.append(idx_to_cls[idx])
                
        data = [self.embedding_list, self.name_list]
        torch.save(data, 'model.pt')
        print(f'model saved as model.pt')
        
        
    def __call__(self, img_path):
        
        img = Image.open(img_path)
        
        img_cropped_list, prob_list = self.mtcnn(img, return_prob=True)
        
        emb = self.resnet(img_cropped_list[0].unsqueeze(0)).detach()
        
        dist_list = []
        
        for idx, emb_db in enumerate(self.embedding_list):
            dist = torch.dist(emb,emb_db).item()
            dist_list.append(dist)

        min_dist = min(dist_list)
        min_dist_idx = dist_list.index(min_dist)
        
        name = self.name_list[min_dist_idx] 
        return name
    
        
