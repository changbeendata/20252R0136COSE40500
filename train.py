"""
Text Classifier Training Module for TELEClass (Phase 3)
Trains classifier using MPNet embeddings + BM25 features with Hierarchical Loss
"""

import os
import json
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from taxonomy import Taxonomy


# ==================== BM25 Vectorizer ====================

class BM25Vectorizer:
    """BM25-style TF-IDF vectorizer"""
    
    def __init__(self, k1=1.5, b=0.75, max_features=1000):
        self.k1 = k1
        self.b = b
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,  # Use log(tf)
            norm='l2'
        )
        self.fitted = False
    
    def fit(self, texts: List[str]):
        """Fit on corpus"""
        print(f"Fitting BM25 on {len(texts)} documents...")
        self.vectorizer.fit(texts)
        self.fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to BM25 vectors"""
        if not self.fitted:
            raise ValueError("Must fit before transform!")
        return self.vectorizer.transform(texts).toarray()


# ==================== Dataset ====================

class HierarchicalTextDataset(Dataset):
    """Dataset with MPNet embeddings + BM25 features for hierarchical classification."""
    
    def __init__(self, doc_ids: List[str], embeddings: np.ndarray, 
                 bm25_features: np.ndarray, labels: List[List[int]]):
        """
        Initialize dataset.
        
        Args:
            doc_ids: List of document IDs
            embeddings: MPNet embeddings [n_docs, embedding_dim]
            bm25_features: BM25 features [n_docs, bm25_dim]
            labels: List of label lists for each document
        """
        self.doc_ids = doc_ids
        self.embeddings = embeddings
        self.bm25_features = bm25_features
        self.labels = labels
        
        assert len(doc_ids) == len(embeddings) == len(bm25_features) == len(labels)
        
        print(f"Dataset initialized with {len(self.doc_ids)} samples")
        print(f"  - MPNet embedding dim: {embeddings.shape[1]}")
        print(f"  - BM25 feature dim: {bm25_features.shape[1]}")
        print(f"  - Total input dim: {embeddings.shape[1] + bm25_features.shape[1]}")
    
    def __len__(self):
        return len(self.doc_ids)
    
    def __getitem__(self, idx):
        return {
            'embedding': torch.tensor(self.embeddings[idx], dtype=torch.float32),
            'bm25': torch.tensor(self.bm25_features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'doc_id': self.doc_ids[idx]
        }


def collate_fn(batch):
    """Custom collate function for variable number of labels."""
    embeddings = torch.stack([item['embedding'] for item in batch])
    bm25 = torch.stack([item['bm25'] for item in batch])
    doc_ids = [item['doc_id'] for item in batch]
    
    # Labels can have different lengths, so we keep them as list
    labels = [item['labels'] for item in batch]
    
    return {
        'embedding': embeddings,
        'bm25': bm25,
        'labels': labels,
        'doc_ids': doc_ids
    }


# ==================== Model ====================

class HierarchicalTextClassifier(nn.Module):
    """MPNet + BM25 multi-label hierarchical classifier."""
    
    def __init__(self, embedding_dim: int, bm25_dim: int, num_classes: int = 531, 
                 hidden_dim: int = 512, dropout: float = 0.3):
        """
        Initialize classifier.
        
        Args:
            embedding_dim: MPNet embedding dimension
            bm25_dim: BM25 feature dimension
            num_classes: Number of classes in taxonomy
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.bm25_dim = bm25_dim
        self.num_classes = num_classes
        
        # Combined input: MPNet + BM25
        input_dim = embedding_dim + bm25_dim
        
        # MLP classifier with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, embedding, bm25):
        """
        Forward pass.
        
        Args:
            embedding: MPNet embeddings [batch_size, embedding_dim]
            bm25: BM25 features [batch_size, bm25_dim]
        
        Returns:
            Logits [batch_size, num_classes]
        """
        # Concatenate features
        x = torch.cat([embedding, bm25], dim=1)
        logits = self.classifier(x)
        return logits
    
    def save(self, path: str):
        """Save model"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'bm25_dim': self.bm25_dim,
            'num_classes': self.num_classes
        }, path)
    
    @classmethod
    def load(cls, path: str, device='cpu'):
        """Load model"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            checkpoint['embedding_dim'],
            checkpoint['bm25_dim'],
            checkpoint['num_classes']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model


# ==================== Loss Functions ====================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits [batch_size, num_classes]
            targets: binary targets [batch_size, num_classes]
        """
        # Apply sigmoid
        probs = torch.sigmoid(inputs)
        
        # Compute focal weight
        # For positive samples: (1 - p)^gamma
        # For negative samples: p^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        # Apply alpha balancing
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HierarchicalLoss(nn.Module):
    """Hierarchical loss that considers taxonomy structure."""
    
    def __init__(self, taxonomy, num_classes: int, 
                 alpha: float = 0.25, gamma: float = 2.0,
                 hierarchy_weight: float = 0.3,
                 root_weight: float = 3.0):
        """
        Args:
            taxonomy: Taxonomy object
            num_classes: Total number of classes
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            hierarchy_weight: Weight for hierarchical consistency loss
            root_weight: Extra weight for root nodes
        """
        super().__init__()
        self.taxonomy = taxonomy
        self.num_classes = num_classes
        self.hierarchy_weight = hierarchy_weight
        
        # Build class weights (higher for roots, balanced for frequency)
        self.class_weights = self._compute_class_weights(root_weight)
        
        # Focal loss for main classification
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        
        # Parent-child relationship matrix
        self.parent_child_matrix = self._build_hierarchy_matrix()
    
    def _compute_class_weights(self, root_weight: float) -> torch.Tensor:
        """Compute per-class weights based on hierarchy."""
        weights = torch.ones(self.num_classes)
        
        for class_id, node in self.taxonomy.nodes.items():
            # Root nodes get extra weight
            if node.parent_id is None:
                weights[class_id] = root_weight
            # Intermediate nodes get moderate weight
            elif len(node.children_ids) > 0:
                weights[class_id] = 2.0
            # Leaf nodes get base weight
            else:
                weights[class_id] = 1.0
        
        return weights
    
    def _build_hierarchy_matrix(self) -> torch.Tensor:
        """Build matrix encoding parent-child relationships."""
        # matrix[i, j] = 1 if i is ancestor of j
        matrix = torch.zeros(self.num_classes, self.num_classes)
        
        for class_id in range(self.num_classes):
            if class_id in self.taxonomy.nodes:
                ancestors = self.taxonomy.get_ancestors(class_id)
                for ancestor_id in ancestors:
                    matrix[ancestor_id, class_id] = 1.0
        
        return matrix
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size, num_classes] binary
        
        Returns:
            Total loss
        """
        device = logits.device
        self.class_weights = self.class_weights.to(device)
        self.parent_child_matrix = self.parent_child_matrix.to(device)
        
        # 1. Weighted focal loss for main classification
        focal_loss = self.focal_loss(logits, targets)
        
        # Apply class weights
        class_weights_expanded = self.class_weights.unsqueeze(0)  # [1, num_classes]
        weighted_loss = focal_loss * class_weights_expanded.mean()
        
        # 2. Hierarchical consistency loss
        # If a child is predicted, its ancestors should also be predicted
        probs = torch.sigmoid(logits)  # [batch, num_classes]
        
        # For each sample, check parent-child consistency
        # parent_probs[i, j] = max probability among ancestors of class j
        parent_probs = torch.matmul(probs, self.parent_child_matrix.t())  # [batch, num_classes]
        
        # Consistency loss: if child has high prob, parent should too
        # Loss = max(0, child_prob - parent_prob)
        consistency_loss = torch.relu(probs - parent_probs).mean()
        
        # 3. Total loss
        total_loss = weighted_loss + self.hierarchy_weight * consistency_loss
        
        return total_loss


# ==================== Training ====================

def create_multi_label_targets(labels_list: List[torch.Tensor], num_classes: int, 
                               device: torch.device) -> torch.Tensor:
    """
    Convert list of label tensors to multi-hot encoding.
    
    Args:
        labels_list: List of label tensors (variable length)
        num_classes: Total number of classes
        device: Device to create tensor on
    
    Returns:
        Multi-hot tensor (batch_size, num_classes)
    """
    batch_size = len(labels_list)
    targets = torch.zeros(batch_size, num_classes, device=device)
    
    for i, labels in enumerate(labels_list):
        for label in labels:
            if 0 <= label < num_classes:
                targets[i, label] = 1.0
    
    return targets


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer, 
               scheduler, device: torch.device, num_classes: int,
               criterion=None, use_hierarchical_loss: bool = True) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    # Use provided criterion or default BCE
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        embedding = batch['embedding'].to(device)
        bm25 = batch['bm25'].to(device)
        labels = batch['labels']
        
        # Create multi-hot targets
        targets = create_multi_label_targets(labels, num_classes, device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(embedding, bm25)
        
        # Compute loss
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device,
            num_classes: int, threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        
        # Create multi-hot targets
        targets = create_multi_label_targets(labels, num_classes, device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        # Compute loss
        loss = criterion(logits, targets)
        total_loss += loss.item()
        
        # Predictions
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Compute metrics
    metrics = {
        'loss': total_loss / len(dataloader),
        'micro_f1': f1_score(all_targets, all_preds, average='micro', zero_division=0),
        'macro_f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'micro_precision': precision_score(all_targets, all_preds, average='micro', zero_division=0),
        'micro_recall': recall_score(all_targets, all_preds, average='micro', zero_division=0),
    }
    
    return metrics


def train_classifier(train_corpus_path: str, train_labels_path: str,
                    val_corpus_path: Optional[str] = None,
                    val_labels_path: Optional[str] = None,
                    model_name: str = "distilbert-base-uncased",
                    num_classes: int = 531,
                    max_length: int = 128,
                    batch_size: int = 16,
                    num_epochs: int = 5,
                    learning_rate: float = 2e-5,
                    warmup_ratio: float = 0.1,
                    output_dir: str = "models/teleclass",
                    device: Optional[str] = None,
                    use_hierarchical_loss: bool = True,
                    focal_alpha: float = 0.25,
                    focal_gamma: float = 2.0,
                    hierarchy_weight: float = 0.3,
                    root_weight: float = 3.0,
                    taxonomy_path: Optional[str] = None) -> nn.Module:
    """
    Train hierarchical text classifier.
    
    Args:
        train_corpus_path: Path to training corpus
        train_labels_path: Path to training pseudo-labels
        val_corpus_path: Path to validation corpus (optional)
        val_labels_path: Path to validation labels (optional)
        model_name: Hugging Face model name
        num_classes: Number of classes
        max_length: Maximum sequence length
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio for scheduler
        output_dir: Directory to save model
        device: Device to train on
        use_hierarchical_loss: Use hierarchical loss with focal loss
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
        hierarchy_weight: Weight for hierarchical consistency
        root_weight: Extra weight for root nodes
        taxonomy_path: Path to enriched taxonomy (for hierarchical loss)
    
    Returns:
        Trained model
    """
    print("=" * 60)
    print("PHASE 3: CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"\nUsing device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = HierarchicalTextDataset(
        train_corpus_path, train_labels_path, tokenizer, max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = None
    if val_corpus_path and val_labels_path:
        val_dataset = HierarchicalTextDataset(
            val_corpus_path, val_labels_path, tokenizer, max_length
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # Initialize model
    print(f"\nInitializing model: {model_name}")
    model = HierarchicalTextClassifier(model_name, num_classes)
    model.to(device)
    
    # Setup optimizer and scheduler
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    
    # Setup loss function
    print("\nSetting up loss function...")
    if use_hierarchical_loss and taxonomy_path and os.path.exists(taxonomy_path):
        print("  Using Hierarchical Loss with:")
        print(f"    - Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        print(f"    - Root node weight: {root_weight}x")
        print(f"    - Hierarchy consistency weight: {hierarchy_weight}")
        
        from taxonomy import Taxonomy
        taxonomy = Taxonomy()
        taxonomy.load_enriched(taxonomy_path)
        
        criterion = HierarchicalLoss(
            taxonomy=taxonomy,
            num_classes=num_classes,
            alpha=focal_alpha,
            gamma=focal_gamma,
            hierarchy_weight=hierarchy_weight,
            root_weight=root_weight
        )
    else:
        print("  Using standard BCEWithLogitsLoss")
        criterion = nn.BCEWithLogitsLoss()
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Training steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, num_classes,
            criterion=criterion, use_hierarchical_loss=use_hierarchical_loss
        )
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        if val_loader:
            val_metrics = evaluate(model, val_loader, device, num_classes)
            print(f"Validation metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Micro F1: {val_metrics['micro_f1']:.4f}")
            print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
            print(f"  Precision: {val_metrics['micro_precision']:.4f}")
            print(f"  Recall: {val_metrics['micro_recall']:.4f}")
            
            # Save best model
            if val_metrics['micro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['micro_f1']
                print(f"  → New best model! Saving to {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                tokenizer.save_pretrained(output_dir)
    
    # Save final model
    print(f"\nSaving final model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    tokenizer.save_pretrained(output_dir)
    
    print("\n✓ Training complete!")
    
    return model


# ==================== Inference ====================

@torch.no_grad()
def predict(model: nn.Module, texts: List[str], tokenizer,
           device: torch.device, max_length: int = 128,
           top_k: int = 3, threshold: float = 0.3) -> List[List[int]]:
    """
    Make predictions on new texts.
    
    Args:
        model: Trained model
        texts: List of text strings
        tokenizer: Tokenizer
        device: Device
        max_length: Maximum sequence length
        top_k: Maximum number of labels to return per document
        threshold: Probability threshold
    
    Returns:
        List of predicted label lists
    """
    model.eval()
    
    predictions = []
    
    # Process in batches
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        
        # Get top-k predictions for each sample
        for prob in probs:
            # Strategy: Always take top-k with highest probabilities
            # This ensures we always predict multiple labels
            top_values, top_indices = torch.topk(prob, k=min(top_k, len(prob)))
            
            # Filter by threshold, but ensure at least 1 prediction
            above_threshold = top_indices[top_values > threshold]
            
            if len(above_threshold) == 0:
                # Take at least top-1, or top-2 if second is reasonably high
                if len(top_indices) >= 2 and top_values[1] > 0.2:
                    predictions.append(top_indices[:2].cpu().tolist())
                else:
                    predictions.append([top_indices[0].item()])
            else:
                predictions.append(above_threshold.cpu().tolist())
    
    return predictions


if __name__ == "__main__":
    # Example: Train on pseudo-labeled data
    model = train_classifier(
        train_corpus_path="Amazon_products/train/train_corpus.txt",
        train_labels_path="pseudo_labels_train.json",
        val_corpus_path=None,  # Add validation set if available
        val_labels_path=None,
        model_name="distilbert-base-uncased",
        num_classes=531,
        batch_size=16,
        num_epochs=3,
        output_dir="models/teleclass"
    )
