import os
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from files.newpipeline import DNAPipeline  
from files.processor import GeneDataProcessor

def assemble_and_export(pred_tsv, predictions_dict):
    # Step 1: Load original TSV
    df = pd.read_csv(pred_tsv, sep="\t")

    # Step 2: Add predictions from dictionary
    df['prediction'] = df['id'].map(predictions_dict)

    # Step 3: Save to new TSV
    df.to_csv("predictions.tsv", sep="\t", index=False)

def main():
    # Data paths
    fasta_file = "data/sequences.fasta"
    labels_tsv = "data/train.tsv"
    pred_tsv = "data/test.tsv"
    
    # Output paths
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data and Create train/val/test split
    print("Loading sequences and labels and Creating data splits...")
    processor = GeneDataProcessor(fasta_file=fasta_file, labels_tsv=labels_tsv, pred_tsv=pred_tsv)
    processor.load_or_process_all()
    sequences = processor.sequences
    labels = processor.labels
    pred_set = processor.pred_set
    train_set, val_set, test_set, test_labels = processor.train_set, processor.val_set, processor.test_set, processor.test_labels
    print(f"Loaded {len(sequences)} sequences and {len(labels)} labels and {len(pred_set)} prediction set")    
    print(f"Created splits: {len(train_set)} training, {len(val_set)} validation, {len(test_set)} test")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = DNAPipeline()
    
    # Train classifier
    print("Training classifier...")
    pipeline.train_classifier(
        training_set=train_set,
        validation_set=val_set,
        batch_size=4,
        learning_rate=5e-4,
        max_epochs=50,
        patience=10,
        warmup_epochs=5,
        accumulation_steps=4
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    results = pipeline.evaluate(
        test_set=test_set,
        ground_truth=test_labels,
        output_file=os.path.join(results_dir, "evaluation_results.json")
    )
    
    print("Evaluation complete!")
    print(f"Overall F1 Score: {results['overall_metrics']['f1']:.4f}")
    print(f"Overall Accuracy: {results['overall_metrics']['accuracy']:.4f}")
    
    # Process a single example in detail to demonstrate sliding window
    if len(test_set) > 0:
        example_gene = next(iter(test_set.keys()))
        example_seq = test_set[example_gene]
        
        print(f"\nDetailed prediction example for gene {example_gene}:")
        print(f"Sequence length: {len(example_seq)}")
        
        # Predict with different window sizes
        for window_size in [4096, 8192, 16384]:
            print(f"\nPredicting with window size {window_size}:")
            pred = pipeline._predict_with_sliding_window(
                example_seq, example_gene, window_size=window_size, overlap=1024
            )
            if example_gene in test_labels:
                true_label = test_labels[example_gene][:len(pred)]
                # Calculate metrics
                tp = sum(1 for p, t in zip(pred, true_label) if p == '1' and t == '1')
                fp = sum(1 for p, t in zip(pred, true_label) if p == '1' and t == '0')
                fn = sum(1 for p, t in zip(pred, true_label) if p == '0' and t == '1')
                tn = sum(1 for p, t in zip(pred, true_label) if p == '0' and t == '0')
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                accuracy = (tp + tn) / len(pred)
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"F1 Score: {f1:.4f}")
            
            # Print a small sample
            sample_start = min(1000, len(example_seq) // 2)
            sample_end = min(sample_start + 50, len(example_seq))
            print("Sample sequence:", example_seq[sample_start:sample_end])
            print("Sample prediction:", pred[sample_start:sample_end])
            if example_gene in test_labels:
                print("Sample true label:", true_label[sample_start:sample_end])
    
    # Process predictions and save to TSV
    print("Processing predictions and saving to TSV...")
    predictions_dict = pipeline.predict(pred_set)
    assemble_and_export(pred_tsv, predictions_dict)
    print("Predictions saved to predictions.tsv")

    print("\nAll done!")

if __name__ == "__main__":
    main()