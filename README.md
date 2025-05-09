# intron-exon-predictor
Predict which parts of your DNA sequence (gene) are introns and exons using this custom ML pipeline (HyenaDNA + fine-tuned Forwardfeeding NN)!

## Data Processing Flow:
***os*** and ***json** handle file I/O
***numpy*** processes the raw data
***torch.utils.data*** prepares it for the model

## Model Training Flow:
***transformers*** provides the pre-trained HyenaDNA model for tokenization and embedding
***torch.nn*** builds our classifier on top of those embeddings
***torch.optim*** optimizes the model parameters
***tqdm*** shows progress
***matplotlib*** visualizes results

## Inference Flow:
***torch*** runs the model
***numpy*** processes the outputs
***json*** saves the results
