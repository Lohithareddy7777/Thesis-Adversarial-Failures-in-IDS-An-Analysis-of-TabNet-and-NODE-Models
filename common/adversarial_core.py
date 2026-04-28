import numpy as np
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from typing import Tuple

from common.utils import get_feature_bounds, clip_to_bounds


class AdversarialAttacker:
    
    def __init__(self, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    def _compute_gradient(self, model: nn.Module, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> torch.Tensor:
        outputs = model(X_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = nn.CrossEntropyLoss()(outputs, y_tensor)
        model.zero_grad()
        loss.backward()
        return X_tensor.grad.data
    
    def fgsm_attack(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.1,
        clip_bounds: Tuple[np.ndarray, np.ndarray] = None
    ) -> np.ndarray:
        print(f"Generating FGSM adversarial examples (epsilon={epsilon})...")
        
        model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        X_tensor.requires_grad = True
        
        gradients = self._compute_gradient(model, X_tensor, y_tensor)
        X_adv = X_tensor + epsilon * gradients.sign()
        
        X_adv = X_adv.detach().cpu().numpy()
        
        if clip_bounds is not None:
            min_bounds, max_bounds = clip_bounds
            X_adv = clip_to_bounds(X_adv, min_bounds, max_bounds)
        
        print(f"Generated {len(X_adv)} adversarial examples")
        return X_adv

    def pgd_attack(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_iter: int = 10,
        random_start: bool = True,
        clip_bounds: Tuple[np.ndarray, np.ndarray] = None,
    ) -> np.ndarray:
        print(f"Generating PGD adversarial examples (epsilon={epsilon}, alpha={alpha}, iters={num_iter})...")

        model.eval()
        X_orig = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        if random_start:
            delta = torch.empty_like(X_orig).uniform_(-epsilon, epsilon)
            X_adv = X_orig + delta
        else:
            X_adv = X_orig.clone()

        if clip_bounds is not None:
            min_bounds, max_bounds = clip_bounds
            min_t = torch.FloatTensor(min_bounds).to(self.device)
            max_t = torch.FloatTensor(max_bounds).to(self.device)
            X_adv = torch.max(torch.min(X_adv, max_t), min_t)

        for _ in range(num_iter):
            X_adv = X_adv.detach().requires_grad_(True)
            gradients = self._compute_gradient(model, X_adv, y_tensor)
            X_adv = X_adv + alpha * gradients.sign()
            X_adv = torch.max(torch.min(X_adv, X_orig + epsilon), X_orig - epsilon)

            if clip_bounds is not None:
                X_adv = torch.max(torch.min(X_adv, max_t), min_t)

        X_adv_np = X_adv.detach().cpu().numpy()
        print(f"Generated {len(X_adv_np)} adversarial examples")
        return X_adv_np
    
    def fgsm_attack_tabnet(
        self,
        model: TabNetClassifier,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.1,
        clip_bounds: Tuple[np.ndarray, np.ndarray] = None
    ) -> np.ndarray:
        print(f"Generating FGSM adversarial examples for TabNet (epsilon={epsilon})...")
        
        network = model.network
        return self.fgsm_attack(network, X, y, epsilon, clip_bounds)

    def pgd_attack_tabnet(
        self,
        model: TabNetClassifier,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_iter: int = 10,
        random_start: bool = True,
        clip_bounds: Tuple[np.ndarray, np.ndarray] = None,
    ) -> np.ndarray:
        network = model.network
        return self.pgd_attack(network, X, y, epsilon, alpha, num_iter, random_start, clip_bounds)

    def _resolve_attack_method(self, attack_type: str, tabnet: bool = False):
        attack = attack_type.upper()
        if attack == "FGSM":
            return self.fgsm_attack_tabnet if tabnet else self.fgsm_attack
        if attack == "PGD":
            return self.pgd_attack_tabnet if tabnet else self.pgd_attack
        raise ValueError(f"Unsupported attack_type: {attack_type}")

    def generate_attack(
        self,
        attack_type: str,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float,
        clip_bounds: Tuple[np.ndarray, np.ndarray] = None,
        alpha: float = 0.1,
        num_iter: int = 10,
    ) -> np.ndarray:
        attack = attack_type.upper()
        if attack == "FGSM":
            return self.fgsm_attack(model, X, y, epsilon=epsilon, clip_bounds=clip_bounds)
        attack_fn = self._resolve_attack_method(attack, tabnet=False)
        return attack_fn(model, X, y, epsilon=epsilon, alpha=alpha, num_iter=num_iter, clip_bounds=clip_bounds)

    def generate_attack_tabnet(
        self,
        attack_type: str,
        model: TabNetClassifier,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float,
        clip_bounds: Tuple[np.ndarray, np.ndarray] = None,
        alpha: float = 0.1,
        num_iter: int = 10,
    ) -> np.ndarray:
        attack = attack_type.upper()
        if attack == "FGSM":
            return self.fgsm_attack_tabnet(model, X, y, epsilon=epsilon, clip_bounds=clip_bounds)
        attack_fn = self._resolve_attack_method(attack, tabnet=True)
        return attack_fn(model, X, y, epsilon=epsilon, alpha=alpha, num_iter=num_iter, clip_bounds=clip_bounds)
    
    def bounded_perturbation_attack(
        self,
        X: np.ndarray,
        epsilon: float = 0.5,
        perturbation_ratio: float = 0.1,
        clip_bounds: Tuple[np.ndarray, np.ndarray] = None
    ) -> np.ndarray:
        print(f"Generating bounded perturbation attack "
              f"(epsilon={epsilon}, ratio={perturbation_ratio})...")
        
        X_adv = X.copy()
        n_samples, n_features = X.shape
        
        if clip_bounds is None:
            min_bounds, max_bounds = get_feature_bounds(X)
        else:
            min_bounds, max_bounds = clip_bounds
        
        n_perturb = int(n_features * perturbation_ratio)
        
        for i in range(n_samples):
            perturb_indices = np.random.choice(n_features, n_perturb, replace=False)
            
            perturbations = np.random.uniform(-epsilon, epsilon, n_perturb)
            
            X_adv[i, perturb_indices] += perturbations
        
        X_adv = clip_to_bounds(X_adv, min_bounds, max_bounds)
        
        print(f"Generated {len(X_adv)} perturbed examples")
        return X_adv