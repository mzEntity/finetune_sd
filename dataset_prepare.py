import h5py
import pandas as pd

from PIL import Image
import os
import json

from datasets import Dataset, Image
from huggingface_hub import login
import os
import json
from utils import read_json_to_dict

def read_json_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_dict_to_json(data, file_path, indent=4):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


def get_dataset(dataset_original_path, image_size=192):
    
    file_path = os.path.join(dataset_original_path, f"UTKFace_{image_size}x{image_size}.h5")

    with h5py.File(file_path, 'r') as f:
        images = f["images"][:]     # (14723, 3, height, width)
        genders = f["genders"][:]   # gender label
        ages = f["labels"][:]       # age label
        races = f["races"][:]       # race label
        
    item_count = images.shape[0]
    
    image_list = []

    for i in range(item_count):
        img_array = images[i].transpose(1, 2, 0)
        img = Image.fromarray(img_array)
        image_list.append(img)
        
    df = pd.DataFrame({'gender': genders, 'image': image_list, 'age': ages, 'race': races})
    
    return df


def spread_local(dataset_original_path, dataset_spread_path, image_size=192):
    df = get_dataset(dataset_original_path, image_size)
    
    meta_list = []
    
    img_dir = os.path.join(dataset_spread_path, f"UTKFace_{image_size}")
    item_count = df.shape[0]
    
    for index, row in df.iterrows():
        file_name = f"{index}.png"
        path = os.path.join(img_dir, file_name)
        meta_data = {
            "idx": index,
            "file": file_name,
            "gender": row['gender'],
            "race": row['race'],
            "age": row['age']
        }
        
        img = row['image']
        img.save(path)
        
        meta_list.append(meta_data)
        print(f"save {index+1}/{item_count}: {path}")
        
    save_dict_to_json(meta_list, os.path.join(img_dir, "config.json"))
    

def upload(dataset_spread_path, image_size, huggingface_token, huggingface_dataset_name):
    image_dir = os.path.join(dataset_spread_path, f"UTKFace_{image_size}")
    
    if not os.path.exists(image_dir):
        print(f"{image_dir} 路径不存在")
        exit()
    
    meta = read_json_to_dict(os.path.join(image_dir, "config.json"))

    image_paths = []
    prompts = []

    for meta_data in meta:
        path = meta_data["path"]
        age = meta_data["age"]
        prompt = f"a portrait of the face of a {age}-year-old" # only use age info
        
        image_paths.append(path)
        prompts.append(prompt)

    login(token=huggingface_token)

    dataset = Dataset.from_dict({
        "image": image_paths,
        "text": prompts
    }).cast_column("image", Image())

    dataset.push_to_hub(huggingface_dataset_name)
    
    
if __name__ == "__main__":
    dataset_original_path = "dataset/h5"
    dataset_spread_path = "dataset/images"
    image_size = 192
    spread_local(dataset_original_path, dataset_spread_path, image_size)
    
    huggingface_token = "hf..."
    huggingface_dataset_name = f"username/UTKFace_{image_size}"
    upload(dataset_spread_path, image_size, huggingface_token, huggingface_dataset_name)