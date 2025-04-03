import torchvision
import torchvision.transforms as transforms
import PIL


def load_model():
    """
    Load a pre-trained Mask R-CNN model with a ResNet-50 backbone.

    This function initializes the Mask R-CNN model pre-trained on the COCO dataset 
    and sets it to evaluation mode for inference.

    Returns:
        torchvision.models.detection.MaskRCNN: The pre-trained Mask R-CNN model.
    """
    #loading pre-trained model. Using Mask R-CNN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    #setting the model to evaluation mode for inference
    model.eval()

    return model


def load_data(image_path):
    """
    Load an image from the specified path and convert it to a PyTorch tensor.

    Args:
        image_path (str): The file path to the image to be loaded.

    Returns:
        torch.Tensor: The image represented as a normalized tensor of shape (C, H, W).
    """
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
    """
     Predict the number of vehicles in an image using a pre-trained Mask R-CNN model.

    This function loads a pre-trained Mask R-CNN model, processes the input image,
    and performs inference to count the number of vehicles detected in the image.

    Args:
        image_path (str): The file path to the input image.
        cuda (bool, optional): Whether to use GPU for inference. Defaults to False.

    Returns:
        int: The number of vehicles detected in the image.
    
    Notes:
        - A detection threshold of 0.5 is applied to filter predictions based on confidence scores.
        - The function specifically counts objects belonging to classes defined in `vehicles_indices`.
    """
    
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