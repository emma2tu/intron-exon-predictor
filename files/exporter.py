import pandas as pd
import numpy as np

def assemble_and_export(test_df, predictions):
    df = test_df.copy()
    df['predictions'] = predictions
    df.to_csv('predictions.csv', sep='\t', index=False)
    return df