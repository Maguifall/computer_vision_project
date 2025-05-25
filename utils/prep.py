import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data(train_dir='data/training', test_dir='data/testing', target_size=(64, 64), batch_size=32):
    """
    Charge les données d'entraînement et de test depuis les dossiers spécifiés.

    Args:
        train_dir (str): Chemin du dossier contenant les données d'entraînement.
        test_dir (str): Chemin du dossier contenant les données de test.
        target_size (tuple): Taille des images à redimensionner.
        batch_size (int): Nombre d'images par batch.

    Returns:
        tuple: ((train_loader, test_loader), (tf_train_generator, tf_test_generator))
    """
    # Transformations PyTorch avec augmentation
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),  # Flip horizontal aléatoire
        transforms.RandomRotation(20),  # Rotation aléatoire jusqu'à 20 degrés
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Ajustements de luminosité/contraste
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Générateurs TensorFlow avec augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    tf_train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    tf_test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return (train_loader, test_loader), (tf_train_generator, tf_test_generator)