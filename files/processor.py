import os
import random
import pandas as pd
from Bio import SeqIO
import joblib

class GeneDataProcessor:
    def __init__(self, fasta_file=None, labels_tsv=None, pred_tsv=None):
        self.fasta_file = fasta_file
        self.labels_tsv = labels_tsv
        self.pred_tsv = pred_tsv
        self.sequences = {}
        self.labels = {}
        self.pred_set = {}
        self.train_set = {}
        self.val_set = {}
        self.test_set = {}
        self.test_labels = {}
        self.cache_path = "cache/dnaprocessor.pkl"
    
    def __str__(self):
        summary = (
            "GeneDataProcessor Summary:\n"
            f"Fasta File: {self.fasta_file}\n"
            f"Labels TSV: {self.labels_tsv}\n"  
            f"Pred TSV: {self.pred_tsv}\n"
            f"Total Sequences: {len(self.sequences)}\n"
            f"Total Labels: {len(self.labels)}\n"
            f"Total Pred Set: {len(self.pred_set)}\n"
            f"Total Train Set: {len(self.train_set)}\n"
            f"Total Val Set: {len(self.val_set)}\n"
            f"Total Test Set: {len(self.test_set)}\n"
            f"Total Test Labels: {len(self.test_labels)}\n"
            f"Cache Path: {self.cache_path}\n"
        )
        return summary

    def process_fasta(self):
        """
        Processes the fasta file and returns a dictionary with gene_id as keys and sequences as values
        """
        print(f"Loading FASTA file: {self.fasta_file}")
        self.sequences = {}

        for i, record in enumerate(SeqIO.parse(self.fasta_file, "fasta")):
            gene_id = record.id
            self.sequences[gene_id] = str(record.seq)
            if i < 3:
                print(f"Found gene: {gene_id}")

        print(f"Total genes loaded: {len(self.sequences)}")
        return self.sequences
    
    def process_labels(self):
        """
        Processes the labels tsv file and returns a dictionary with gene_id as keys and labels as values
        """
        print(f"Loading labels TSV file: {self.labels_tsv}")
        self.labels = {}
        
        label_df = pd.read_csv(self.labels_tsv, sep="\t")
        
        for i, row in label_df.iterrows():
            gene_id = row['id']
            label_string = row['label']
            self.labels[gene_id] = label_string
        
        print(f"Total labels loaded: {len(self.labels)}")
        return self.labels
    
    def create_pred_set(self):
        """
        Creates the prediction set with only sequences as values
        """
        self.pred_set = {}
        
        pred_df = pd.read_csv(self.pred_tsv, sep="\t")
        
        for i, row in pred_df.iterrows():
            gene_id = row['id']
            sequence = self.sequences[gene_id]
            self.pred_set[gene_id] = sequence
        
        return self.pred_set
    
    def create_train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, random_seed=42, max_labeled_genes=30000):
        """
        Split the data into training, validation, and test sets
        """
        # Find genes that have both sequence and labels
        common_genes = sorted(set(self.sequences.keys()) & set(self.labels.keys()))
        
            # Optionally limit total number of labeled genes
        if max_labeled_genes is not None:
            common_genes = common_genes[:max_labeled_genes]
            print(f"Limiting to first {max_labeled_genes} labeled genes for training/validation/testing.")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        random.shuffle(common_genes)
        
        # Calculate split indices
        train_end = int(len(common_genes) * train_ratio)
        val_end = train_end + int(len(common_genes) * val_ratio)
        
        # Split gene IDs
        train_genes = common_genes[:train_end]
        val_genes = common_genes[train_end:val_end]
        test_genes = common_genes[val_end:]
        
        # Create datasets
        self.train_set = {gene_id: (self.sequences[gene_id], self.labels[gene_id]) for gene_id in train_genes}
        self.val_set = {gene_id: (self.sequences[gene_id], self.labels[gene_id]) for gene_id in val_genes}
        self.test_set = {gene_id: self.sequences[gene_id] for gene_id in test_genes}
        self.test_labels = {gene_id: self.labels[gene_id] for gene_id in test_genes}
        
        return self.train_set, self.val_set, self.test_set, self.test_labels
        
    def load_or_process_all(self, fasta_file=None, labels_tsv=None, test_tsv=None, max_labeled_genes=30000):
        """
        Load or process all data
        """
        if fasta_file:
            self.fasta_file = fasta_file
        if labels_tsv:
            self.labels_tsv = labels_tsv
        if test_tsv:
            self.test_tsv = test_tsv
        
        # Try loading from cache
        cached = self.load_cache(self.cache_path)
        if cached:
            return cached
        
        # Process everything from scratch
        print("Processing from scratch...")
        self.process_fasta()
        self.process_labels()
        self.create_pred_set()
        self.create_train_val_test_split(max_labeled_genes=max_labeled_genes)
        self.save_cache() # Save to cache
        
        return self
    
    def save_cache(self):
        """
        Saves the processor object to a cache file
        """
        if not os.path.exists("cache"):
            os.makedirs("cache")
        
        with open(self.cache_path, 'wb') as f:
            joblib.dump(self, f)

    @staticmethod
    def load_cache(cache_path="cache/dnaprocessor.pkl"):
        """
        Loads the GeneDataProcessor object from a cache file.
        """
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                print(f"✅ Cache loaded from: {cache_path}")
                return joblib.load(f)
        else:
            print("⚠️ Cache file not found.")
            return None
