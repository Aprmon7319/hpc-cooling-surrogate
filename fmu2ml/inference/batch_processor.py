"""
Batch inference processor for efficient predictions.
"""
import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from .predictor import CoolingModelPredictor


class BatchProcessor:
    """Process multiple predictions in batches"""
    
    def __init__(self, predictor: CoolingModelPredictor, batch_size: int = 32):
        self.predictor = predictor
        self.batch_size = batch_size
    
    def process_batch(self, inputs: List[np.ndarray]) -> List[Dict]:
        """
        Process a list of inputs in batches
        
        Parameters:
        -----------
        inputs : List[np.ndarray]
            List of input sequences
        
        Returns:
        --------
        List[Dict] : List of predictions
        """
        results = []
        
        for i in tqdm(range(0, len(inputs), self.batch_size), desc="Processing batches"):
            batch = inputs[i:i + self.batch_size]
            batch_tensor = torch.stack([torch.FloatTensor(x) for x in batch])
            
            with torch.no_grad():
                predictions = self.predictor.predict(batch_tensor)
                if isinstance(predictions, dict):
                    results.append(predictions)
                else:
                    results.extend(predictions)
        
        return results