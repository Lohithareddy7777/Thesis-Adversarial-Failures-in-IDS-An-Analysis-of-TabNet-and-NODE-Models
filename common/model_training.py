import numpy as np
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from typing import Tuple
import os

from common.utils import set_random_seed, save_model, print_section_header


class NODEClassifier(nn.Module):
    
    def __init__(self, input_dim: int, num_classes: int = 2, num_layers: int = 3, 
                 layer_dim: int = 64):
        super(NODEClassifier, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            current_dim = layer_dim
        
        layers.append(nn.Linear(layer_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_embeddings(self, x):
        embeddings = []
        current = x
        
        for layer in self.network[:-1]:
            current = layer(current)
            if isinstance(layer, nn.ReLU):
                embeddings.append(current.detach())
        
        return embeddings


class ModelTrainer:
    
    def __init__(self, random_state: int = 42, device: str = None):
        self.random_state = random_state
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        set_random_seed(random_state)
        
        print(f"Using device: {self.device}")
    
    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        unique, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(unique) * counts)
        return weights / weights.sum() * len(unique)
    
    def train_tabnet(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int = 2,
        max_epochs: int = 50,
        patience: int = 10
    ) -> TabNetClassifier:
        print_section_header("TRAINING TABNET MODEL")
        
        class_weights = self._compute_class_weights(y_train)
        sample_weights = np.array([class_weights[label] for label in y_train])
        
        model = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=4,
            gamma=1.3,
            n_independent=1,
            n_shared=1,
            seed=self.random_state,
            verbose=0,
            device_name=self.device,
            mask_type='sparsemax'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=512,
            virtual_batch_size=64,
            eval_metric=['accuracy'],
            weights=sample_weights
        )
        
        print("TabNet training complete!")
        return model
    
    def train_node(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int = 2,
        max_epochs: int = 50,
        patience: int = 10,
        batch_size: int = 1024,
        learning_rate: float = 0.001
    ) -> NODEClassifier:
        print_section_header("TRAINING NODE MODEL")
        
        input_dim = X_train.shape[1]
        model = NODEClassifier(input_dim, num_classes).to(self.device)
        
        class_weights = self._compute_class_weights(y_train)
        class_weights_t = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(max_epochs):
            model.train()
            
            indices = torch.randperm(len(X_train_t))
            X_train_t = X_train_t[indices]
            y_train_t = y_train_t[indices]
            
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i+batch_size].to(self.device)
                batch_y = y_train_t[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_acc = train_correct / train_total
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_acc = (val_predicted == y_val_t).sum().item() / len(y_val_t)
            
            scheduler.step(val_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{max_epochs} - "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"NODE training complete! Best validation accuracy: {best_val_acc:.4f}")
        return model
    
    def predict_tabnet(self, model: TabNetClassifier, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = model.predict(X)
        probas = model.predict_proba(X)
        return preds, probas
    
    def predict_node(self, model: NODEClassifier, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = model(X_t)
            probas = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probas, axis=1)
        
        return preds, probas
    
    def save_models(self, tabnet_model, node_model, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        tabnet_path = os.path.join(output_dir, 'tabnet_model.pkl')
        save_model(tabnet_model, tabnet_path)
        print(f"Saved TabNet model to: {tabnet_path}")
