from torchvision import transforms

# Prediction preprocess
preprocess_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

preprocess_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3))], p=0.3),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]
                                 )
])


def get_transforms(resize=(256, 256), single_channel=False, training=True, prediction=False):
    # Create list of transformations
    transfom_list = list()

    # Transform to pil
    if prediction:
        transfom_list.append(transforms.ToPILImage())

    # Resize
    transfom_list.append(transforms.Resize(resize))

    # Transformations for training
    if training:
        transfom_list.append(transforms.RandomHorizontalFlip())
        transfom_list.append(transforms.RandomVerticalFlip())
        transfom_list.append(transforms.RandomRotation(degrees=30))
        transfom_list.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3))], p=0.3))

    # Grayscale
    if single_channel:
        transfom_list.append(transforms.Grayscale(num_output_channels=1))
    else:
        transfom_list.append(transforms.Grayscale(num_output_channels=3))

    # To tensor
    transfom_list.append(transforms.ToTensor())

    # Normalize
    if single_channel:
        transfom_list.append(transforms.Normalize(mean=[0.485], std=[0.229]))
    else:
        transfom_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # Compose transformations
    TRANSFORM = transforms.Compose(transfom_list)

    return TRANSFORM
