import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from models.cnn import CNN1
from models.tf_cnn import create_cnn2

# 📈 Fonction pour afficher et sauvegarder les courbes loss + accuracy
def plot_metrics(train_losses, train_accuracies, model_name):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Loss', color='red')
    plt.plot(epochs, train_accuracies, label='Accuracy', color='blue')
    plt.title(f'Training Loss and Accuracy for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_training_plot.png")
    plt.show()

# 🚂 Fonction d'entraînement
def train(model, data_loader, epochs, device=None, framework="pytorch", learning_rate=0.001):
    if framework == "pytorch":
        if not isinstance(model, nn.Module):
            raise ValueError("Le modèle PyTorch doit être une instance de nn.Module")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Taux d'apprentissage ajustable
        train_loader = data_loader
        train_losses = []
        train_accuracies = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = epoch_loss / len(train_loader)
            acc = correct / total
            train_losses.append(avg_loss)
            train_accuracies.append(acc)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

        torch.save(model.state_dict(), "MAGATTE_FALL_model.torch")
        print(f"✅ Modèle sauvegardé sous MAGATTE_FALL_model.torch")
        return {'loss': train_losses, 'accuracy': train_accuracies}

    elif framework == "tensorflow":
        if not hasattr(model, 'fit'):
            raise ValueError("Le modèle TensorFlow doit être un modèle Keras")
        train_loader, test_loader = data_loader
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_loader, epochs=epochs, validation_data=test_loader, verbose=1)
        model.save("MAGATTE_FALL_model.tensorflow")
        print(f"✅ Modèle sauvegardé sous MAGATTE_FALL_model.tensorflow")
        return history

# 🔬 Fonction d'évaluation
def evaluate(model, test_loader, device=None, framework="pytorch"):
    if framework == "pytorch":
        if not isinstance(model, nn.Module):
            raise ValueError("Le modèle PyTorch doit être une instance de nn.Module")
        model.eval()
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = test_loss / len(test_loader)
        accuracy = correct / total
        print(f"🔍 Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    elif framework == "tensorflow":
        if not hasattr(model, 'evaluate'):
            raise ValueError("Le modèle TensorFlow doit être un modèle Keras")
        loss, accuracy = model.evaluate(test_loader, verbose=0)
        print(f"🔍 Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")
        return loss, accuracy