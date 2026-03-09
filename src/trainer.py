import argparse

import sklearn
from sklearn.datasets import load_wine, load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.metrics import balanced_accuracy_score, accuracy_score

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import time
import pandas as pd

# De momento solo tenemos una arquitecutra de red, para ampliar, se tendría que mover a otro archivo todos los modelos (por dejarlo limpio)
# Luego se llamarian parecidos a la de SUPPORTED_DATASETS, con un nombre clave (ej. "my_model": MyModel) y se podrían elegir con un argumento de línea de comandos (ej. --model my_model)
# La más posible problematica sería que algunos modelos requieran argumentos específicos que se añadirían al parser y se pasan a lo que corresponda (se puede ir viendo)
# Otra problematica sería que ciertos modelos requieran un procesamiento específico de los datos o incluso de datasets específicos (se va observando también, pero de momento no es el caso con esta arquitectura tan simple)
class Trainer(nn.Module):

    def __init__(self, layersDims, batchNormalization=False):
        super(Trainer, self).__init__()
        self.layersDims = layersDims
        self.layers = nn.Sequential()
        self.batchNormalization = batchNormalization

        if (len(layersDims) < 2):
            raise ValueError("layersDims must have at least 2 elements (input and output dimensions)")

        if (len(layersDims) > 2):
            for i in range(len(layersDims) - 2):
                self.layers.add_module(f"linear_{i}", nn.Linear(layersDims[i], layersDims[i + 1], bias=True))
                if self.batchNormalization:
                    self.layers.add_module(f"batch_norm_{i}", nn.BatchNorm1d(layersDims[i + 1]))
                self.layers.add_module(f"relu_{i}", nn.ReLU())

        self.layers.add_module(f"linear_{len(layersDims) - 2}", nn.Linear(layersDims[-2], layersDims[-1], bias=True))

        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

# Añadir aqui como se cargan los datasets, por ejemplo con sklearn.datasets.load_iris() o similar
# Si varian mucho de los de juguete, como mínimo tienen que ser de la forma [X, y], donde X en un array de shape (n_samples, n_features) y y un array de shape (n_samples,) con las etiquetas de clase
# Si se quiere añadir un dataset nuevo, simplemente hay que añadir una nueva función de carga y añadirla al diccionario SUPPORTED_DATASETS con un nombre clave (ej. "my_dataset": load_my_dataset) 
SUPPORTED_DATASETS = {
    "wine": load_wine,
    "iris": load_iris,
    "digits": load_digits,
    "breast_cancer": load_breast_cancer
}

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # Squeeze outputs for binary classification (BCEWithLogitsLoss)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                outputs = outputs.squeeze()
                targets = targets.float()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        # Validation step
        model.eval()
        val_loss = 0
        accuracy = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # Squeeze outputs for binary classification (BCEWithLogitsLoss)
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    outputs = outputs.squeeze()
                    targets = targets.float()
                    loss = criterion(outputs, targets)
                    predicted = (torch.sigmoid(outputs) > 0.5).long()
                else:
                    loss = criterion(outputs, targets)
                    _, predicted = torch.max(outputs, 1)
                val_loss += loss.item()
                accuracy += (predicted == targets.long()).sum().item()

        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {accuracy / len(val_loader.dataset):.4f}", flush=True)
    
    return total_loss / len(train_loader)

def main(args):

    seed = args.seed
    torch.manual_seed(seed)
    sklearn.utils.check_random_state(seed)
    bathcNormailization = args.batch_normalization


    load_data = SUPPORTED_DATASETS[args.dataset]
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=seed, stratify=data.target)  # Stratify to maintain class distribution in train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=seed, stratify=y_train)  # 0.25 x 0.8 = 0.2

    print("Empezando entrenamiento del modelo completo para clasificación total...")
    time_inicio = time.time()
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model for full classification task (input layer + hidden layers + output layer)
    model = Trainer(layersDims=[X_train.shape[1]] + args.hidden_layers + [len(data.target_names)], batchNormalization=bathcNormailization)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    model.to(device)

    train(model, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer, device, epochs=50)
    time_fin = time.time()
    tiempo_entrenamiendo_completo = time_fin - time_inicio
    print(f"\nTiempo total de entrenamiento del modelo completo: {tiempo_entrenamiendo_completo:.4f} segundos", flush=True)

    binary_models = []
    print("Empezando entrenamiento de modelos binarios para clasificación parcial...")
    tiempo_inicio = time.time()
    # Models for patial classification tasks (input layer + hidden layers + output layer 1 class, acepting only samples of that class)
    for class_idx in range(len(data.target_names)):
        print(f"\nTraining model for class '{data.target_names[class_idx]}'")
        binary_y_train = (y_train == class_idx).astype(int)
        binary_y_val = (y_val == class_idx).astype(int)

        binary_train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(binary_y_train, dtype=torch.long))
        binary_val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(binary_y_val, dtype=torch.long))

        binary_train_loader = DataLoader(binary_train_dataset, batch_size=args.batch_size, shuffle=True)
        binary_val_loader = DataLoader(binary_val_dataset, batch_size=args.batch_size, shuffle=False)

        binary_model = Trainer(layersDims=[X_train.shape[1]] + args.hidden_layers + [1], batchNormalization=bathcNormailization)
        binary_model.to(device)
        train(binary_model, binary_train_loader, binary_val_loader, nn.BCEWithLogitsLoss(), optim.Adam(binary_model.parameters(), lr=0.001), device, epochs=50)
        binary_models.append(binary_model)
    tiempo_fin = time.time()
    tiempo_entrenamiendo_binarios = tiempo_fin - tiempo_inicio
    print(f"\nTiempo total de entrenamiento de modelos binarios: {tiempo_entrenamiendo_binarios:.4f} segundos", flush=True)

    ### Test the full classification model
    model.eval()
    
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    
    test_loss = 0
    acc_test = 0
    acc_weighted_test = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            test_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    acc_weighted_test = balanced_accuracy_score(all_labels, all_preds)
    acc_test = accuracy_score(all_labels, all_preds)
    print(f"\nTest Loss for full classification model: {test_loss / len(test_loader):.4f}", flush=True)
    print(f"Test Accuracy for full classification model: {acc_test:.4f}", flush=True)
    print(f"Test Weighted Accuracy for full classification model: {acc_weighted_test:.4f}", flush=True)

    ### Test the binary classification models
    for class_idx, binary_model in enumerate(binary_models):
        binary_model.eval()
        binary_test_loss = 0
        binary_acc_test = 0
        binary_y_test = (y_test == class_idx).astype(int)
        binary_test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(binary_y_test, dtype=torch.long))
        binary_test_loader = DataLoader(binary_test_dataset, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in binary_test_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                binary_targets = (targets == 1).long()
                outputs = binary_model(inputs).squeeze()
                loss = nn.BCEWithLogitsLoss()(outputs, binary_targets.float())
                binary_test_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).long()
                binary_acc_test += (predicted == binary_targets).sum().item()

        print(f"\nTest Loss for binary model of class '{data.target_names[class_idx]}': {binary_test_loss / len(binary_test_loader):.4f}", flush=True)
        print(f"Test Accuracy for binary model of class '{data.target_names[class_idx]}': {binary_acc_test / len(binary_test_dataset):.4f}", flush=True)

    ### Test the ensemble of binary classification models
    ensemble_acc_test = 0
    ensemble_outputs_all = []
    ensemble_labels_all = []
    
    print("\n=== DEBUGGING ENSEMBLE PREDICTIONS ===")
    sample_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            binary_outputs = []
            for binary_model in binary_models:
                binary_output = torch.sigmoid(binary_model(inputs).squeeze())
                binary_outputs.append(binary_output)
            ensemble_outputs = torch.stack(binary_outputs, dim=1)
            
            # Ensemble strategy: threshold-based voting with fallback
            threshold = 0.5
            batch_predictions = []
            
            for sample_idx in range(ensemble_outputs.shape[0]):
                sample_probs = ensemble_outputs[sample_idx]  # Probabilidades para esta muestra
                true_label = targets[sample_idx].item()
                
                # Debug: imprimir las primeras 5 muestras
                if sample_count < 5:
                    print(f"\nSample {sample_count + 1} (True label: {true_label} - {data.target_names[true_label]}):")
                    for i, prob in enumerate(sample_probs):
                        print(f"  Model {i} ({data.target_names[i]}): {prob:.4f}")
                
                # Ver qué modelos superan el threshold
                above_threshold = sample_probs > threshold
                
                if above_threshold.sum() == 1:
                    # Solo un modelo supera threshold -> esa clase
                    predicted_class = above_threshold.nonzero(as_tuple=True)[0].item()
                    strategy = "single_confident"
                elif above_threshold.sum() > 1:
                    # Varios superan threshold -> el de mayor probabilidad entre los que superan
                    candidates = sample_probs * above_threshold.float()
                    predicted_class = candidates.argmax().item()
                    strategy = "multiple_confident"
                else:
                    # Ninguno supera threshold -> el de mayor probabilidad (fallback)
                    predicted_class = sample_probs.argmax().item()
                    strategy = "fallback"
                
                if sample_count < 5:
                    print(f"  Strategy: {strategy}")
                    print(f"  Predicted: {predicted_class} ({data.target_names[predicted_class]})")
                    print(f"  Correct: {'✅' if predicted_class == true_label else '❌'}")
                
                batch_predictions.append(predicted_class)
                sample_count += 1
            
            ensemble_outputs_all.extend(batch_predictions)
            ensemble_labels_all.extend(targets.cpu().numpy())
    
    print("=====================================\n")

    ensemble_acc_weighted_test = balanced_accuracy_score(ensemble_labels_all, ensemble_outputs_all)
    ensemble_acc_test = accuracy_score(ensemble_labels_all, ensemble_outputs_all)
    print(f"Ensemble Test Accuracy: {ensemble_acc_test:.4f}", flush=True)
    print(f"Ensemble Test Weighted Accuracy: {ensemble_acc_weighted_test:.4f}", flush=True)

    df = pd.DataFrame({
        "Dataset": args.dataset,
        "Number of Classes": len(data.target_names),
        "Hidden Layers": str(args.hidden_layers),
        "Batch Normalization": args.batch_normalization,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "Full Model Train Time (s)": tiempo_entrenamiendo_completo,
        "Binary Models Train Time (s)": tiempo_entrenamiendo_binarios,
        "Full Model Test Accuracy": acc_test,
        "Full Model Test Weighted Accuracy": acc_weighted_test,
        "Ensemble Test Accuracy": ensemble_acc_test,
        "Ensemble Test Weighted Accuracy": ensemble_acc_weighted_test
    }, index=[0])

    df.to_csv("training_results.csv", index=False, mode="a", header=not pd.io.common.file_exists("training_results.csv"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a simple feedforward neural network on a dataset")
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATASETS.keys(), default="wine", help="The dataset to use for training")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[], help="The dimensions of the hidden layers (e.g. --hidden-layers 64 32 for two hidden layers with 64 and 32 units respectively)")
    parser.add_argument("-s", "--seed", type=int, default=2000, help="Random seed for reproducibility")
    parser.add_argument("--batch-normalization", action="store_true", help="Whether to use batch normalization in the hidden layers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation")


    args = parser.parse_args()

    main(args)