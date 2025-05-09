import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import time
import json
from datetime import datetime

# Constants
HYENA_MODEL_NAME = 'LongSafari/hyenadna-small-32k-seqlen-hf'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class IntronExonTokenClassifier(nn.Module):
    """
    A classifier that takes per-base embeddings and outputs per-base predictions.
    Uses bidirectional LSTM for context-aware predictions. 
    Uses feedforward classifier head for per-token binary classification (e.g. intron vs exon).
    """
    def __init__(self, input_size=256, hidden_size=128, num_layers=2, dropout=0.2):
        super(IntronExonTokenClassifier, self).__init__()
        
        # Bidirectional LSTM to capture context in both directions
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer (bidirectional LSTM has 2*hidden_size features)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Ensures output is between 0 and 1 for binary classification
        )
    
    def forward(self, x, lengths=None):
        """
        Forward pass for sequence classification
        Args:
            x: Tensor of shape [batch_size, seq_len, input_size]
            lengths: Optional tensor of sequence lengths for packed sequence processing
        Returns:
            Tensor of shape [batch_size, seq_len, 1] with probabilities for each token
        """
        # If we have lengths, we can use packed sequences for efficiency
        if lengths is not None:
            # Pack the sequences into real (unpadded) lengths
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            
            # Process packed sequence with LSTM
            packed_output, _ = self.lstm(packed_x)
            
            # Unpack the sequences to get back padded shape
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            # Process without packing (when all sequences have same length)
            lstm_out, _ = self.lstm(x)
        
        # Apply classifier to each token's hidden state
        return self.classifier(lstm_out)
    

class DNASequenceDataset(Dataset):
    """
    Custom dataset containing pairs of DNA sequence per-token embeddings and their intron/exon labels.
    """
    def __init__(self, embeddings, labels=None):
        """
        Args:
            embeddings: List of numpy arrays where each array is [seq_len, embedding_dim]
            labels: List of numpy arrays where each array is [seq_len] with 0/1 values
        """
        self.embeddings = embeddings
        self.labels = labels
        self.is_test = labels is None
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if self.is_test:
            return torch.FloatTensor(self.embeddings[idx])
        else:
            return (
                torch.FloatTensor(self.embeddings[idx]), 
                torch.FloatTensor(self.labels[idx]).unsqueeze(-1)  # Add dimension for BCE loss
            )
        


class WarmupScheduler:
    """
    Learning rate scheduler with warm-up phase followed by decay.
    The learning rate increases linearly during the warm-up phase and then decays exponentially for the remainder of training.    
    """
    def __init__(self, optimizer, warmup_steps, max_lr, min_lr=1e-6, decay_factor=0.95):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.current_step = 0
        self._setup_initial_lr()
    
    def _setup_initial_lr(self):
        # Start with a very small learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
    
    def step(self):
        """Update learning rate based on current step"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warm-up phase
            progress = self.current_step / self.warmup_steps
            lr = self.min_lr + progress * (self.max_lr - self.min_lr)
        else:
            # Exponential decay phase
            steps_after_warmup = self.current_step - self.warmup_steps
            lr = max(self.max_lr * (self.decay_factor ** steps_after_warmup), self.min_lr)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class DNAPipeline:
    def __init__(self, model_name=HYENA_MODEL_NAME, device=None):
        """
        Initializes the DNAPipeline for per-base intron/exon classification.
        """
        self._set_device(device)
        self._set_model(model_name)
        self._set_classifier()
        self._set_embedding_cache_dir()
        self._setup_dirs()
        print("Pipeline initialized.")

    def _set_device(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def _set_model(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        print(f"Model set to: {model_name}")

    def _set_classifier(self, classifier=None):
        # Using the improved token classifier with correct input size
        self.classifier = classifier or IntronExonTokenClassifier(
            input_size=256, hidden_size=128).to(self.device)
        self.classifier_path = "cache/classifier/best_classifier.pth"
        print(f"Classifier set to: IntronExonTokenClassifier with Bidirectional LSTM")       

    def _set_embedding_cache_dir(self, embedding_cache_dir=None):
        self.embedding_cache_dir = embedding_cache_dir or "cache/embeddings/"
        print(f"Embedding cache directory set to: {self.embedding_cache_dir}")

    def _setup_dirs(self):
        """Create necessary directories"""
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.classifier_path), exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def get_embedding(self, sequence, gene_id=None, use_sliding_window=True, 
                      window_size=8192, overlap=1024):
        """
        Return per-base embeddings for a given DNA sequence.
        For long sequences, uses sliding window approach.
        
        Args:
            sequence: DNA sequence string
            gene_id: Optional gene identifier for caching
            use_sliding_window: Whether to use sliding window for long sequences
            window_size: Size of sliding window
            overlap: Overlap between windows
            
        Returns:
            numpy array of shape [seq_len, embedding_dim]
        """
        MAX_SEQ_LEN = 32000  # Slightly below model max to be safe
        
        # Check if the sequence is already cached

        if gene_id:
            npz_path = os.path.join(self.embedding_cache_dir, f"{gene_id}.npz")

            if os.path.exists(npz_path):
                try:
                    data = np.load(npz_path)["embedding"]
                    if data.size == 0:
                        raise ValueError(f"Empty embedding in {npz_path}, skipping.")
                    return data
                except Exception as e:
                    print(f"Failed to load {npz_path}: {e}")

        # For shorter sequences, compute embeddings directly
        if len(sequence) <= MAX_SEQ_LEN:
            result = self._compute_embedding_direct(sequence)
        else:
            # For longer sequences, use sliding window
            if use_sliding_window:
                print(f"Sequence {gene_id if gene_id else ''} length {len(sequence)} exceeds model limit. Using sliding window.")
                result = self._compute_embedding_sliding_window(sequence, window_size, overlap)
            else:
                print(f"Truncating {gene_id if gene_id else ''} from {len(sequence)} to {MAX_SEQ_LEN} bases")
                sequence = sequence[:MAX_SEQ_LEN]
                result = self._compute_embedding_direct(sequence)
            
        # Save the embeddings to cache if gene_id is provided
        if gene_id:
            cache_file = os.path.join(self.embedding_cache_dir, f"{gene_id}.npz")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            if result is None or result.size == 0 or not isinstance(result, np.ndarray):
                # Delete any existing cache file to avoid keeping corrupted data
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                raise ValueError(f"Cannot save embedding for {gene_id} — invalid format: {type(result)}")
            else:
                print(f"Saving embedding to: {cache_file}")
                print(f"Embedding shape: {result.shape} — dtype: {result.dtype}")
                np.savez_compressed(cache_file, embedding=result)

        return result
    
    def _compute_embedding_direct(self, sequence):
        """Compute embeddings for a sequence directly"""
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)
        return embeddings.detach().cpu().numpy()
    
    def _compute_embedding_sliding_window(self, sequence, window_size=8192, overlap=1024):
        """
        Compute embeddings using sliding window approach for long sequences
        
        Args:
            sequence: DNA sequence string
            window_size: Size of each window
            overlap: Overlap between consecutive windows
            
        Returns:
            numpy array of embeddings for the full sequence
        """
        stride = window_size - overlap
        embeddings_list = []
        
        # Process each window
        for start_idx in range(0, len(sequence), stride):
            # Extract window with overlap
            end_idx = min(start_idx + window_size, len(sequence))
            window = sequence[start_idx:end_idx]
            
            # Skip small windows at the end
            if len(window) < 32:  # Minimum sequence length
                continue
                
            # Compute embeddings for this window
            window_embedding = self._compute_embedding_direct(window)
            
            # For the first window, keep everything
            if start_idx == 0:
                embeddings_list.append(window_embedding)
            else:
                # For subsequent windows, skip the overlapping part
                embeddings_list.append(window_embedding[overlap:])
                
            # Break if we've processed the entire sequence
            if end_idx >= len(sequence):
                break
        
        # Concatenate all embeddings
        if not embeddings_list:
            raise ValueError("Failed to generate embeddings for sequence")
            
        return np.concatenate(embeddings_list, axis=0)
   
    def prepare_dataset(self, data_dict, is_training=True):
        """
        Prepare embeddings and labels from sequences
        
        Args:
            data_dict: Dictionary of {gene_id: (sequence, labels)} for training/validation
                       or {gene_id: sequence} for testing
            is_training: Whether this is for training (with labels) or testing (no labels)
            
        Returns:
            Dataset object
        """
        embeddings, labels = [], []
        MAX_TRAIN_VAL_EMBEDS = 30000
        
        # Process each gene
        embedded_count = 0
        for gene_id, data in tqdm(data_dict.items(), desc="Processing sequences"):
            if embedded_count >= MAX_TRAIN_VAL_EMBEDS:
                print(f"🛑 Reached embed limit of {MAX_TRAIN_VAL_EMBEDS}, stopping early.")
                break
            
            if is_training:
                seq, label_str = data
                # Convert label string to numeric array
                # Get embeddings and append to list
                label_array = np.array([int(char) for char in label_str], dtype=np.float32)
                embedding = self.get_embedding(seq, gene_id=gene_id)
                if embedding is None or embedding.size == 0:
                    print(f"Warning: No embedding for {gene_id}, skipping.")
                    continue

                if embedding.shape[0] == len(label_array) + 1:
                    embedding = embedding[:-1]  # Remove the last token (special token at end)
                elif embedding.shape[0] != len(label_array):
                    print(f"Skipping gene {gene_id}: Embedding length {embedding.shape[0]} != label length {len(label_array)}")
                    continue
                
                embeddings.append(embedding)
                labels.append(label_array)
                embedded_count += 1
                # print(f"Processed {gene_id}: Embedding shape {embedding.shape}, Label shape {label_array.shape}")

            else:
                seq = data  # For test data, we only have sequences
                # Get embeddings and append to list
                embedding = self.get_embedding(seq, gene_id=gene_id)
                if embedding is None or embedding.size == 0:
                    print(f"Warning: No embedding for {gene_id}, skipping.")
                    continue
                embeddings.append(embedding)
                embedded_count += 1
            
        
        # Create dataset
        if is_training:
            return DNASequenceDataset(embeddings, labels)
        else:
            return DNASequenceDataset(embeddings)
    
    def collate_fn(self, batch):
        """
        Custom collate function to handle variable length sequences
        """
        if isinstance(batch[0], tuple):  # Training data with labels
            # Separate embeddings and labels
            embeddings = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            
            # Get lengths before padding
            lengths = torch.tensor([emb.size(0) for emb in embeddings])
            
            # Pad sequences to same length
            padded_embeddings = pad_sequence(embeddings, batch_first=True)
            padded_labels = pad_sequence(labels, batch_first=True)
            
            return padded_embeddings, padded_labels, lengths
        else:  # Test data without labels
            lengths = torch.tensor([emb.size(0) for emb in batch])
            padded_embeddings = pad_sequence(batch, batch_first=True)
            return padded_embeddings, lengths
    
    def get_data_loader(self, dataset, batch_size=8, shuffle=True):
        """
        Create DataLoader with appropriate collate function for variable length sequences
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=min(4, torch.get_num_threads() // 2) if not self.device.type == 'cuda' else 0,
            pin_memory=self.device.type == 'cuda'
        )
    
    def train_classifier(self, training_set, validation_set=None, batch_size=4, 
                         learning_rate=5e-4, max_epochs=100, patience=10, 
                         warmup_epochs=5, accumulation_steps=4):
        """
        Train the classifier with all optimizations:
        - Gradient accumulation
        - Learning rate warm-up
        - Early stopping
        - Best model saving
        
        Args:
            training_set: Dictionary of {gene_id: (sequence, labels)}
            validation_set: Optional dictionary of {gene_id: (sequence, labels)}
            batch_size: Batch size for training
            learning_rate: Maximum learning rate after warm-up
            max_epochs: Maximum number of epochs to train
            patience: Number of epochs with no improvement before early stopping
            warmup_epochs: Number of epochs for warm-up
            accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Trained classifier
        """
        # Start training run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/training_{run_id}.log"
        results_file = f"results/training_{run_id}.json"
        
        # Log initial information
        with open(log_file, 'w') as f:
            f.write(f"Training run started at {datetime.now()}\n")
            f.write(f"Model: {HYENA_MODEL_NAME}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Learning rate: {learning_rate}\n")
            f.write(f"Accumulation steps: {accumulation_steps}\n")
            f.write(f"Effective batch size: {batch_size * accumulation_steps}\n")
            f.write(f"Warmup epochs: {warmup_epochs}\n")
            f.write(f"Max epochs: {max_epochs}\n")
            f.write(f"Patience: {patience}\n")
            f.write(f"Training set size: {len(training_set)}\n")
            if validation_set:
                f.write(f"Validation set size: {len(validation_set)}\n")
            
        print(f"Starting training run {run_id}")
        print(f"Log file: {log_file}")
        
        # Prepare datasets
        start_time = time.time()
        train_dataset = self.prepare_dataset(training_set, is_training=True)
        print(f"🔍 Final training set size: {len(train_dataset)}")
        if len(train_dataset) == 0:
            raise ValueError("Training set is empty! Cannot create DataLoader.")
        train_loader = self.get_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"Training dataset prepared in {time.time() - start_time:.2f} seconds")
        
        if validation_set:
            start_time = time.time()
            val_dataset = self.prepare_dataset(validation_set, is_training=True)
            val_loader = self.get_data_loader(val_dataset, batch_size=batch_size, shuffle=False)
            print(f"Validation dataset prepared in {time.time() - start_time:.2f} seconds")

        # Training setup
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        
        # Calculate warmup steps
        warmup_steps = len(train_loader) * warmup_epochs // accumulation_steps
        scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps, max_lr=learning_rate)
        
        # Early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model = None
        best_epoch = 0

        # For learning curves
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []
        learning_rates = []

        # Training loop
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            
            # Train one epoch
            train_loss, batch_train_metrics = self._train_epoch(
                train_loader, loss_fn, optimizer, scheduler, accumulation_steps)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Track training metrics
            train_losses.append(train_loss)
            train_metrics.append(batch_train_metrics)
            
            # Log epoch information
            epoch_time = time.time() - epoch_start_time
            log_msg = (f"Epoch {epoch+1}/{max_epochs}: "
                      f"Train Loss = {train_loss:.4f}, "
                      f"Train F1 = {batch_train_metrics['f1']:.4f}, "
                      f"LR = {current_lr:.6f}, "
                      f"Time = {epoch_time:.2f}s")
            
            # Validate if validation set provided
            if validation_set:
                val_loss, batch_val_metrics = self._validate(val_loader, loss_fn)
                val_losses.append(val_loss)
                val_metrics.append(batch_val_metrics)
                
                log_msg += (f", Val Loss = {val_loss:.4f}, "
                           f"Val F1 = {batch_val_metrics['f1']:.4f}")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model = copy.deepcopy(self.classifier.state_dict())
                    best_epoch = epoch
                    torch.save(best_model, self.classifier_path)
                    log_msg += f" [IMPROVED - saved model]"
                else:
                    epochs_no_improve += 1
                    log_msg += f" [No improvement for {epochs_no_improve} epochs]"
                    
                if epochs_no_improve >= patience:
                    log_msg += f" [Early stopping triggered]"
                    print(log_msg)
                    with open(log_file, 'a') as f:
                        f.write(log_msg + "\n")
                    break
            
            # Log epoch results
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg + "\n")

        # Training summary
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Load best model if available
        if best_model is not None:
            self.classifier.load_state_dict(best_model)
            print(f"Loaded best model from epoch {best_epoch+1}")
        
        # Save training results
        results = {
            "run_id": run_id,
            "model": HYENA_MODEL_NAME,
            "training_time": training_time,
            "epochs": epoch+1,
            "best_epoch": best_epoch+1,
            "train_losses": train_losses,
            "train_metrics": train_metrics,
            "val_losses": val_losses if validation_set else None,
            "val_metrics": val_metrics if validation_set else None,
            "learning_rates": learning_rates,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_epochs": warmup_epochs,
            "patience": patience,
            "training_set_size": len(training_set),
            "validation_set_size": len(validation_set) if validation_set else 0
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot learning curves
        self._plot_learning_curves(train_losses, val_losses if validation_set else None, 
                                   learning_rates, run_id)
        
        return self.classifier
    
    def _train_epoch(self, train_loader, loss_fn, optimizer, scheduler, accumulation_steps=4):
        """
        Train for one epoch with gradient accumulation
        """
        self.classifier.train()
        total_loss = 0
        num_batches = 0
        
        # For metrics calculation
        all_preds = []
        all_labels = []
        all_masks = []
        
        # Zero gradients at the beginning
        optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack batch
            batch_X, batch_y, lengths = batch_data
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            y_pred = self.classifier(batch_X, lengths)
            
            # Create a mask for padding tokens
            mask = self._create_padding_mask(batch_y, lengths)
            
            # Apply mask and calculate loss only on real tokens
            # Normalize loss by accumulation steps
            loss = self._masked_loss(y_pred, batch_y, mask, loss_fn) / accumulation_steps
            
            # Backward pass (accumulate gradients)
            loss.backward()
            
            # Track predictions and labels for metrics calculation
            all_preds.append((y_pred > 0.5).float().cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            
            # Track statistics (use the unnormalized loss for reporting)
            total_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            # Update weights every accumulation_steps or at the end of the epoch
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                
                # Perform optimization step
                optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Zero gradients for next accumulation
                optimizer.zero_grad()
        
        # Compute overall metrics
        metrics = self._compute_metrics(all_preds, all_labels, all_masks)
        
        return total_loss / num_batches, metrics
    
    def _validate(self, val_loader, loss_fn):
        """
        Validate model and compute metrics
        """
        self.classifier.eval()
        total_loss = 0
        num_batches = 0
        
        # For metrics calculation
        all_preds = []
        all_labels = []
        all_masks = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Unpack batch
                batch_X, batch_y, lengths = batch_data
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                y_pred = self.classifier(batch_X, lengths)
                
                # Create a mask for padding tokens
                mask = self._create_padding_mask(batch_y, lengths)
                
                # Apply mask and calculate loss only on real tokens
                loss = self._masked_loss(y_pred, batch_y, mask, loss_fn)
                
                # Track predictions and labels for metrics calculation
                all_preds.append((y_pred > 0.5).float().cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
                all_masks.append(mask.cpu().numpy())
                
                # Track statistics
                total_loss += loss.item()
                num_batches += 1
        
        # Compute overall metrics
        metrics = self._compute_metrics(all_preds, all_labels, all_masks)
        
        return total_loss / num_batches, metrics
    
    def _create_padding_mask(self, tensor, lengths):
        """
        Create a mask for padding tokens
        """
        batch_size, max_len = tensor.shape[0], tensor.shape[1]
        mask = torch.zeros((batch_size, max_len, 1), device=tensor.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        return mask
    
    def _masked_loss(self, pred, target, mask, loss_fn):
        """
        Apply mask to compute loss only on real tokens
        """
        # Ensure pred and target have the same shape
        if pred.shape != target.shape:
            pred = pred[:, :target.shape[1]]
        
        masked_pred = pred * mask
        masked_target = target * mask
        # Sum loss and divide by number of real tokens
        return loss_fn(masked_pred, masked_target) * mask.sum() / (mask.sum() + 1e-8)
    
    def _compute_metrics(self, all_preds, all_labels, all_masks):
        """
        Compute precision, recall, F1 score
        """
        # Flatten and filter by mask
        preds = []
        labels = []
        
        for pred_batch, label_batch, mask_batch in zip(all_preds, all_labels, all_masks):
            for i in range(pred_batch.shape[0]):  # For each sequence in batch
                seq_len = int(mask_batch[i].sum())
                if seq_len > 0:  # Only include real tokens
                    preds.extend(pred_batch[i, :seq_len, 0])
                    labels.extend(label_batch[i, :seq_len, 0])
        
        preds = np.array(preds).round()
        labels = np.array(labels)
        
        # Compute metrics
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        return {
            'precision': float(precision), 
            'recall': float(recall), 
            'f1': float(f1),
            'accuracy': float(accuracy),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    def _plot_learning_curves(self, train_losses, val_losses=None, learning_rates=None, run_id=None):
        """
        Plot learning curves with multiple subplots
        """
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot losses
        axs[0].plot(train_losses, label='Training Loss')
        if val_losses:
            axs[0].plot(val_losses, label='Validation Loss')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training and Validation Loss')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot learning rate
        if learning_rates:
            axs[1].plot(learning_rates, label='Learning Rate')
            axs[1].set_ylabel('Learning Rate')
            axs[1].set_xlabel('Epoch')
            axs[1].set_title('Learning Rate Schedule')
            axs[1].grid(True)
            axs[1].set_yscale('log')
        
        plt.tight_layout()
        
        # Save figure
        if run_id:
            plt.savefig(f'results/learning_curves_{run_id}.png')
        else:
            plt.savefig('results/learning_curves.png')
        
        plt.close()
    
    def predict(self, test_set, batch_size=4, use_sliding_window=True, 
               window_size=8192, overlap=1024):
        """
        Predict intron/exon labels for test sequences with sliding window
        
        Args:
            test_set: Dictionary of {gene_id: sequence}
            batch_size: Batch size for inference
            use_sliding_window: Whether to use sliding window for long sequences
            window_size: Size of each window
            overlap: Overlap between consecutive windows
            
        Returns:
            Dictionary of {gene_id: label_string}
        """
        # Start time
        start_time = time.time()
        
        # Load best model if available
        if os.path.exists(self.classifier_path):
            self.classifier.load_state_dict(torch.load(self.classifier_path))
            print(f"Loaded trained model from {self.classifier_path}")
        
        # Set model to evaluation mode
        self.classifier.eval()
        
        # Dictionary to store results
        predictions = {}
        
        # Run ID for logging
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/prediction_{run_id}.log"
        
        with open(log_file, 'w') as f:
            f.write(f"Prediction run started at {datetime.now()}\n")
            f.write(f"Model: {HYENA_MODEL_NAME}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Test set size: {len(test_set)}\n")
            f.write(f"Use sliding window: {use_sliding_window}\n")
            f.write(f"Window size: {window_size}\n")
            f.write(f"Overlap: {overlap}\n")
        
        if use_sliding_window:
            # Process each sequence individually with sliding window
            for gene_id, sequence in tqdm(test_set.items(), desc="Predicting with sliding window"):
                predictions[gene_id] = self._predict_with_sliding_window(
                    sequence, gene_id, window_size, overlap)
                
                # Log progress
                if len(predictions) % 10 == 0:
                    with open(log_file, 'a') as f:
                        f.write(f"Processed {len(predictions)}/{len(test_set)} sequences\n")
        else:
            # Use batch processing for shorter sequences
            # Prepare dataset
            test_dataset = self.prepare_dataset(test_set, is_training=False)
            test_loader = self.get_data_loader(test_dataset, batch_size=batch_size, shuffle=False)
            
            gene_ids = list(test_set.keys())
            gene_idx = 0
            
            with torch.no_grad():
                for batch_data in tqdm(test_loader, desc="Predicting in batches"):
                    # Unpack batch
                    batch_X, lengths = batch_data
                    batch_X = batch_X.to(self.device)
                    
                    # Forward pass
                    y_pred = self.classifier(batch_X, lengths)
                    
                    # Convert predictions to binary (0/1)
                    binary_preds = (y_pred > 0.5).float().cpu().numpy()
                    
                    # Store predictions for each sequence
                    for i, length in enumerate(lengths):
                        if gene_idx < len(gene_ids):
                            gene_id = gene_ids[gene_idx]
                            # Convert binary predictions to string of 0s and 1s
                            pred_string = ''.join(map(str, binary_preds[i, :length, 0].astype(int)))
                            predictions[gene_id] = pred_string
                            gene_idx += 1
        
        # Log completion
        prediction_time = time.time() - start_time
        print(f"Prediction completed in {prediction_time:.2f} seconds")
        with open(log_file, 'a') as f:
            f.write(f"Prediction completed in {prediction_time:.2f} seconds\n")
            f.write(f"Total sequences processed: {len(predictions)}\n")
            
        return predictions
    
    def _predict_with_sliding_window(self, sequence, gene_id=None, window_size=8192, overlap=1024):
        """
        Make predictions using a sliding window approach for very long sequences
        
        Args:
            sequence: DNA sequence string
            gene_id: Gene identifier (for logging)
            window_size: Size of each window
            overlap: Overlap between consecutive windows
            
        Returns:
            String of 0s and 1s representing intron/exon predictions
        """
        # For very short sequences, process directly
        if len(sequence) <= window_size:
            embedding = self.get_embedding(sequence, gene_id=gene_id)
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = self.classifier(embedding_tensor).squeeze()
                
            # Convert predictions to binary string
            return ''.join((pred > 0.5).float().cpu().numpy().astype(int).astype(str))
        
        # For long sequences, use sliding window
        stride = window_size - overlap
        predictions = []
        window_start = 0
        
        # Process each window
        for window_idx in range(0, (len(sequence) + stride - 1) // stride):
            # Extract window
            window_start = window_idx * stride
            window_end = min(window_start + window_size, len(sequence))
            window = sequence[window_start:window_end]
            
            # Skip small windows at the end
            if len(window) < 32:  # Minimum sequence length
                continue
                
            # Get embeddings for this window
            embedding = self.get_embedding(window)
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                window_pred = self.classifier(embedding_tensor).squeeze()
            
            # Convert to binary predictions
            binary_pred = (window_pred > 0.5).float().cpu().numpy()
            
            # Only keep non-overlapping part (except for first window)
            if window_idx > 0:
                binary_pred = binary_pred[overlap:]
              
            # Add to predictions
            predictions.extend(binary_pred)
        
        # Make sure we have a prediction for each base in the original sequence
        # (in case our sliding windows didn't cover everything exactly)
        if len(predictions) > len(sequence):
            predictions = predictions[:len(sequence)]
        elif len(predictions) < len(sequence):
            # This shouldn't happen with proper window and stride, but just in case
            missing = len(sequence) - len(predictions)
            predictions.extend([0] * missing)
            print(f"Warning: {missing} bases were not covered by sliding windows for {gene_id}")
        
        # Convert to string
        return ''.join(np.array(predictions).astype(int).astype(str))
    
    def evaluate(self, test_set, ground_truth, output_file=None):
        """
        Evaluate model on test set with ground truth
        
        Args:
            test_set: Dictionary of {gene_id: sequence}
            ground_truth: Dictionary of {gene_id: label_string}
            output_file: Optional file path to save detailed results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        predictions = self.predict(test_set)
        
        # Prepare for evaluation
        all_preds = []
        all_labels = []
        
        # Per-gene metrics
        gene_metrics = {}
        
        # Evaluate each gene
        for gene_id, pred_string in predictions.items():
            if gene_id in ground_truth:
                true_string = ground_truth[gene_id]
                
                # Ensure prediction and ground truth have same length
                min_len = min(len(pred_string), len(true_string))
                pred_string = pred_string[:min_len]
                true_string = true_string[:min_len]
                
                # Convert to binary arrays
                pred_array = np.array([int(char) for char in pred_string])
                true_array = np.array([int(char) for char in true_string])
                
                # Add to overall evaluation
                all_preds.extend(pred_array)
                all_labels.extend(true_array)
                
                # Compute gene-specific metrics
                tp = ((pred_array == 1) & (true_array == 1)).sum()
                fp = ((pred_array == 1) & (true_array == 0)).sum()
                fn = ((pred_array == 0) & (true_array == 1)).sum()
                tn = ((pred_array == 0) & (true_array == 0)).sum()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                
                gene_metrics[gene_id] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'accuracy': float(accuracy),
                    'length': int(min_len),
                    'tp': int(tp),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tn': int(tn)
                }
        
        # Compute overall metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        tn = ((all_preds == 0) & (all_labels == 0)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Aggregate results
        results = {
            'overall_metrics': {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'total_bases': int(len(all_preds))
            },
            'gene_metrics': gene_metrics
        }
        
        # Save detailed results if output file provided
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                
        # Print summary
        print(f"Evaluation Summary:")
        print(f"Total bases: {len(all_preds)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return results


