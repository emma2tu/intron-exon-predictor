from files.processor import GeneDataProcessor
from files.pipeline import DNAPipeline
from files.exporter import assemble_and_export
from config import FASTA, TRAIN_TSV, TEST_TSV

def main(): 
    print("Starting data processing...")
    processor = GeneDataProcessor.load_or_process_all(
        fasta_file=FASTA,
        train_tsv=TRAIN_TSV,
        test_tsv=TEST_TSV
    )
    print("Data processing complete.")

    # pipeline = DNAPipeline()
    # print("Starting model training...")
    # pipeline.train_classifier(processor.train_set, processor.val_set)
    # print("Model training complete.")
    # print("Starting model evaluation...")
    # raw_preds, predictions = pipeline.evaluate(processor.test_set)
    # print("Model evaluation complete.")

    # print("Starting predictions export...")
    # predictions_df = assemble_and_export(
    #     test_df=processor.test_df,
    #     predictions=predictions
    # )
    # print("Predictions exported to predictions.csv")

if __name__ == "__main__":
    main()