import pandas as pd
from Bio import SeqIO
import joblib
import os

class GeneDataProcessor:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.gene_to_sequence_dict = {}
        self.training_set = {}
        self.validation_set = {}
        self.testing_set = {}
        self.cache_path = "cache/processor.pkl"
    
    def __str__(self):
        summary = (
            "GeneDataProcessor Summary:\n"
            f"  Genes in raw FASTA: {len(self.gene_to_sequence_dict)}\n"
            f"  Genes in train.tsv: {len(self.training_set)+len(self.validation_set)}\n"
            f"  Genes in test.tsv: {len(self.testing_set)}\n"
            f"  Training set size: {len(self.training_set)}\n"
            f"  Validation set size: {len(self.validation_set)}\n"
            f"  Testing set size: {len(self.testing_set)}"
        )
        return summary

    def process_fasta(self, fasta_file):
        """
        Processes the fasta file and returns a dictionary with gene_id as keys and sequences as values
        """

        self.gene_to_sequence_dict = {}
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            gene_id = record.id
            self.gene_to_sequence_dict[gene_id] = str(record.seq)
        
        return self.gene_to_sequence_dict

    def process_tsv(self, train_tsv, test_tsv):
        """
        Processes the train and test tsv files and returns dataframes
        """
        self.train_df = pd.read_csv(train_tsv, sep="\t")
        self.test_df = pd.read_csv(test_tsv, sep="\t")
        
        return self.train_df, self.test_df

    def split_train_val(self):
        """
        Splits the train dataframe into training and validation sets
        """
        # Assuming the last 20% of the data is used for validation
        val_size = int(0.2 * len(self.train_df))
        temp_train_data = self.train_df[:-val_size]
        temp_val_data = self.train_df[-val_size:]
        self.train_set = dict(temp_train_data.iloc[:, [0, 1]].values)
        self.val_set = dict(temp_val_data.iloc[:, [0, 1]].values)

        # Convert the set values into tuples of (sequence, label)
        for gene_id in self.train_set.keys():
            sequence = self.gene_to_sequence_dict[gene_id]
            label = self.train_set[gene_id]
            self.train_set[gene_id] = (sequence, label)

        for gene_id in self.val_set.keys():
            sequence = self.gene_to_sequence_dict[gene_id]
            label = self.val_set[gene_id]
            self.val_set[gene_id] = (sequence, label)
        
        return self.train_set, self.val_set

    def create_test_set(self):
        """
        Creates the test set with only sequences as values
        """
        self.test_set = {}
        
        for index, row in self.test_df.iterrows():
            gene_id = row['id']
            sequence = self.gene_to_sequence_dict[gene_id]
            self.test_set[gene_id] = sequence
        
        return self.test_set

    def save_cache(self):
        """
        Saves the processor object to a cache file
        """
        if not os.path.exists("cache"):
            os.makedirs("cache")
        
        with open(self.cache_path, 'wb') as f:
            joblib.dump(self, f)
    @staticmethod

    def load_cache():
        """
        Loads the processor object from a cache file
        """
        if os.path.exists("cache/processor.pkl"):
            with open("cache/processor.pkl", 'rb') as f:
                return joblib.load(f)
        else:
            return None
    @staticmethod

    def load_or_process_all(self, fasta_file, train_tsv, test_tsv):
        """
        Loads the processor object from a cache file or processes all files if cache does not exist
        """
        processor = GeneDataProcessor.load_cache()
        
        if processor is None:
            processor = GeneDataProcessor()
            processor.process_fasta(fasta_file)
            processor.process_tsv(train_tsv, test_tsv)
            processor.split_train_val()
            processor.create_test_set()
            processor.save_cache()
        
        return processor