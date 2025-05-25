import os
import torch
import argparse
import matplotlib.pyplot as plt
from utils.prep import get_data
from models.cnn import CNN1
from models.tf_cnn import create_cnn2
from models.train import train, evaluate, plot_metrics
from tensorflow.keras.models import load_model  # <-- ajout ici

def parse_args():
    parser = argparse.ArgumentParser(description="Entraîner ou tester un modèle")
    parser.add_argument("--model", type=str, default="CNN1", help="CNN1 ou CNN2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mode", type=str, default="train", help="train ou test")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Taux d'apprentissage")
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.model.upper()
    (train_loader, test_loader), (tf_train_generator, tf_test_generator) = get_data()

    if model_name == "CNN1":
        model = CNN1()
        model_path = "MAGATTE_FALL_model.torch"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if args.mode == "train":
            history = train(model, train_loader, args.epochs, device=device, learning_rate=args.learning_rate)
            torch.save(model.state_dict(), model_path)
            plot_metrics(history['loss'], history['accuracy'], model_name)

        elif args.mode == "test":
            model.load_state_dict(torch.load(model_path))
            evaluate(model, test_loader, device=device)

    elif model_name == "CNN2":
        model_path = "MAGATTE_FALL_model.tensorflow"

        if args.mode == "train":
            model = create_cnn2()
            history = train(model, (tf_train_generator, tf_test_generator), args.epochs,
                            framework="tensorflow", learning_rate=args.learning_rate)
            plot_metrics(history.history['loss'], history.history['accuracy'], model_name)

        elif args.mode == "test":
            model = load_model(model_path)  # ✅ Corrigé ici
            evaluate(model, tf_test_generator, framework="tensorflow")

if __name__ == "__main__":
    main()
