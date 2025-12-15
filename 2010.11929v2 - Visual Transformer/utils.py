import collections 
from collections import defaultdict
import json 

def pre_index(annotation_path):
    with open(annotation_path, 'r') as file:
        annotations = json.load(file)

    id_annotations = defaultdict(list)
    id_images = {}
    id_categories = {} 

    for item in annotations["annotations"]:
        img_id = item['image_id']
        id_annotations[img_id].append([item['category_id']] + item['bbox'])

    for item in annotations["images"]:
        id = item['id']
        id_images[id] = (item['file_name'], item['width'], item['height'])
    
    for item in annotations['categories']:
        category_id = item['id']
        id_categories[category_id] = item['name']
        
    return (id_annotations, id_images, id_categories)
