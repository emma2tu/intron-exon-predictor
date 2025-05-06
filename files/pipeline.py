from transformers import AutoTokenizer, AutoModel
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from tqdm import tqdm
from config import HYENA_MODEL_NAME, EMBED_DIM

# IntronExonClassifier Imports
import torch
import torch.nn as nn

class IntronExonClassifier(nn.Module):
    """
    A simple feedforward neural network for binary intron/exon classification at the token (nucleotide) level.
    input: (batch_size, seq_len, embed_dim)
    output: (batch_size, seq_len) — one label per token
    """
    def __init__(self, input_size=EMBED_DIM, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size), # 256 → 128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 64),         # 128 → 64
            nn.ReLU(),
            nn.Linear(64, 1),         # 64 → 1
            nn.Sigmoid()              # Sigmoid activation for binary classification
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        output = self.model(x)       # (batch_size, seq_len, 1)
        return output.squeeze(-1)    # (batch_size, seq_len)


class DNAPipeline:
    def __init__(self, model_name=HYENA_MODEL_NAME, device=None):
        """
        Initializes the DNAPipeline including the device, tokenizer, model, and classifier.
        """
        self._set_device(self, device)
        self._set_model(self, model_name)
        self._set_classifier(self)
        self._set_embedding_cache_dir(self)
        print("Pipeline initialized.")

    def _set_device(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def _set_model(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"Model set to: {model_name}")

    def _set_classifier(self, classifier=None):
        self.scaler = StandardScaler()
        self.classifier = classifier or IntronExonClassifier(input_size=768, hidden_size=128).to(self.device)
        self.classifier_path = "cache/classifier/best_classifier.pth"
        print(f"Classifier set to: IntronExonClassifier as ForwardFeeding Neural Network")       

    def _set_embedding_cache_dir(self, embedding_cache_dir=None):
        self.embedding_cache_dir = embedding_cache_dir or "cache/embeddings/"
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        print(f"Embedding cache directory set to: {self.embedding_cache_dir}")

    def get_embedding(self, sequence, gene_id=None):
        """
        Return per-base embeddings (logits) for a given DNA sequence.
        """
        # Check if the sequence is already cached
        if gene_id:
            cache_file = os.path.join(self.embedding_cache_dir, f"{gene_id}.npy")
            if os.path.exists(cache_file):
                return np.load(cache_file)
        # If not cached, compute the embeddings
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        result = embeddings.detach().cpu().numpy()  # Safely move back to CPU
        # Save the embeddings to cache if gene_id is provided
        if gene_id:
            np.save(cache_file, result)
        return result
   
    def make_dataset_arrays(self, data_set, fit_transform=True):
        X, y = [], []
        for i, (gene_id, (seq, labels)) in enumerate(tqdm(data_set.items(), desc="Embedding training set")):
            embeddings = self.get_embedding(seq, gene_id=gene_id) 
            X.append(embeddings)
            y.append(labels)
        if fit_transform:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return np.array(X), np.array(y)
    
    def make_testset_array(self, test_set):
        X = []
        for gene_id, seq in tqdm(test_set.items(), desc="Embedding test set"):
            embeddings = self.get_embedding(seq, gene_id=gene_id)
            X.append(embeddings)
        X = self.scaler.transform(X)  # Note: use transform, not fit_transform for test data
        return np.array(X)
    
    def get_data_loader(self, X, y=None, batch_size=32, shuffle=True):
        # Convert numpy arrays to torch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        
        # Determine optimal number of workers
        num_workers = min(4, torch.get_num_threads() // 2)

        # Create and return DataLoader
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers if not self.device.type == 'cuda' else 0,  # 0 for CUDA as data is already on GPU
            pin_memory=self.device.type == 'cuda'  # Pin memory if using CUDA
        )
        
    def train_classifier(self, training_set, validation_set=None, batch_size=32, 
                         learning_rate=0.01, max_epochs=100, patience=10, 
                         checkpoint_path="cache/classifier/best_classifier.pth"):
        
        # Data Preparation
        X_train, y_train = self.make_dataset_arrays(training_set)
        train_loader = self.get_data_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        
        if validation_set:
            self.scaler.fit(X_train)  # Ensure scaler is fit only on training data
            X_val, y_val = self.make_dataset_arrays(validation_set, fit_transform=False)
            val_loader = self.get_data_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

        # Training Setup
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        
        # Early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model = None

        # For learning curves
        train_losses = []
        val_losses = []

        # Training Loop
        for epoch in range(max_epochs):
            # Train one epoch
            train_loss = self._train_epoch(train_loader, loss_fn, optimizer)
            train_losses.append(train_loss)

            # Validate if validation set provided
            if validation_set:
                val_loss, val_accuracy = self._validate(val_loader, loss_fn)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch+1}/{max_epochs}: Train Loss = {train_loss:.4f}, " 
                      f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model = copy.deepcopy(self.classifier.state_dict())
                    torch.save(best_model, checkpoint_path)
                    print(f"Model improved. Saved checkpoint to {checkpoint_path}")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epochs")
                    
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                print(f"Epoch {epoch+1}/{max_epochs}: Train Loss = {train_loss:.4f}")

            # Load best model if available
        if best_model is not None:
            self.classifier.load_state_dict(best_model)
            print(f"Loaded best model from epoch {epoch+1 - epochs_no_improve}")
        
        # Plot learning curves if validation was used
        if validation_set:
            self._plot_learning_curves(train_losses, val_losses)
        
        return self.classifier
    
    def _train_epoch(self, train_loader, loss_fn, optimizer):
        self.classifier.train()  # Set model to training mode
        total_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            y_pred = self.classifier(batch_X)
            loss = loss_fn(y_pred, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate(self, val_loader, loss_fn):
        self.classifier.eval()  # Set model to evaluation mode
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0
        with torch.no_grad():  # No gradient computation
            for batch_X, batch_y in val_loader:
                # Forward pass
                y_pred = self.classifier(batch_X)
                loss = loss_fn(y_pred, batch_y)
                
                # Track statistics
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate accuracy (for binary classification)
                predictions = (y_pred > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.numel()
        accuracy = correct / total
        return total_loss / num_batches, accuracy    # Tuple of (validation loss, validation accuracy)

    def evaluate(self, test_set, batch_size=64, threshold=0.5):
        # Prepare test data
        X_test = self.make_testset_array(test_set)
        y_test = None
        
        # Create data loader
        test_loader = self.get_data_loader(X_test, y_test, batch_size=batch_size, shuffle=False)
        
        # Evaluate
        self.classifier.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_X, in test_loader:  # Note the comma after batch_X
                y_pred = self.classifier(batch_X)
                predictions.extend(y_pred.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)

        binary_preds = (predictions > threshold).astype(int)
        return {
            'predictions': predictions,
            'binary_predictions': binary_preds
        }
    
    def save_classifier(self):
        torch.save(self.classifier.state_dict(), self.classifier_path)
        print(f"Classifier saved to {self.classifier_path}")

    def load_classifier(self):
        self.classifier.load_state_dict(torch.load(self.classifier_path, map_location=self.device))
        self.classifier.to(self.device)
        print(f"Classifier loaded from {self.classifier_path}")
        return self.classifier
    
    def _plot_learning_curves(self, train_losses, val_losses):
        """
        Plot learning curves from training.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Learning Curves')
        plt.grid(True)
        plt.savefig('learning_curves.png')
        plt.show()
        print("Learning curves saved to learning_curves.png")

        X_val, y_val = self.make_dataset(validation_set)
        y_pred = self.classifier.predict(X_val)
        print("Validation Predictions: ", y_pred)
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        print("Validation Accuracy:", acc)
        print(report)
        return acc, report
        X_test = self.generate_inputset(testing_set)
        y_pred = self.classifier.predict(X_test)
        return y_pred 
 
        if os.path.exists(self.classifier_path):
            self.model = joblib.load(self.classifier_path)
        else:
            raise FileNotFoundError(f"Model file not found at {self.classifier_path}")