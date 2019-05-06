import torchvision.transforms as transforms


def transform(trans, img_size):
    if trans == "transform2":
        #ImageNet mean, std
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size[0]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        transform_val = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize
        ])
        return {'train': transform_train, 'val': transform_val, 'test': transform_val}
    elif trans == "transform2_2":
        #ImageNet mean, std
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size[0], scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        transform_val = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize
        ])
        return {'train': transform_train, 'val': transform_val, 'test': transform_val}
    
    elif trans == "transform8":
        #ImageNet mean, std
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size[0], scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((-15,15), resample=False, expand=False, center=None),
            transforms.ToTensor(),
            normalize
        ])
        transform_val = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize
        ])
        return {'train': transform_train, 'val': transform_val, 'test': transform_val}
    
    else :
        print("Error : transformation KEY error")
        raise
