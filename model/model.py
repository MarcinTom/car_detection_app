import torchvision
import torchvision.transforms as transforms
import PIL


def load_model():
    #loading pre-trained model. Using Mask R-CNN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    #setting the model to evaluation mode for inference
    model.eval()

    return model


def load_data(image_path):
    #loading PIL Image from path and creating a torch tensor from it
    img_pil = PIL.Image.open(image_path)
    img_tensor = transforms.functional.to_tensor(img_pil)

    return img_tensor


#From PyTorch documentation for the Mask R-CNN classes
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic', 'light', 'fire', 'hydrant', 'N/A', 'stop',
    'sign', 'parking', 'meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'ball',
    'kite', 'baseball', 'bat', 'baseball', 'glove', 'skateboard', 'surfboard', 'tennis',
    'racket', 'bottle', 'N/A', 'wine', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted', 'plant', 'bed', 'N/A', 'dining', 'table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell',
    'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy', 'bear', 'hair', 'drier', 'toothbrush'
]

#class indices corresponding to types of cars
vehicles_indices = [3]

def predict(image_path, cuda = False):
    
    #setting the model to evaluation mode for inference
    model = load_model()

    img_tensor = load_data(image_path)

    #if GPU is wanted (from argument)
    if cuda:
        img_tensor = img_tensor.cuda()
        model.cuda()
    else:
        img_tensor = img_tensor.cpu()
        model.cpu()

    predictions = model([img_tensor])

    n_vehicles = 0

    for i in range(predictions[0]["boxes"].shape[0]):
        #set the threshold for the prediction as you like. Here is 0.5
        if predictions[0]["scores"][i] > 0.5:
            label_id = predictions[0]["labels"][i].item()

            if label_id in vehicles_indices:
                n_vehicles += 1

    return n_vehicles