import torch
import numpy as np
from pathlib import Path

# Import TNO components from your implementation
from fmu2ml.data.processors import (
    TNOSequenceConfig,
    create_tno_data_loaders,
    create_tno_model,
    create_tno_trainer,
    get_sample_batch,
    print_batch_shapes
)



# Also set environment variables
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'

# =========================================================================
# Configuration
# =========================================================================
DATA_PATH = "../summit/data"  # Update this to your data path
CHECKPOINT_DIR = "checkpoints/tno_example"

# Sequence parameters
HISTORY_LENGTH = 30   # L: number of past timesteps
PREDICTION_HORIZON = 10  # K: number of future timesteps to predict

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
PATIENCE = 20

NUM_WORKERS = 32

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =========================================================================
# Step 1: Create Data Loaders
# =========================================================================
print("\n" + "="*60)
print("Step 1: Creating Data Loaders")
print("="*60)

tno_seq_config = TNOSequenceConfig(
    history_length=HISTORY_LENGTH,
    prediction_horizon=PREDICTION_HORIZON,
    stride=1,
    pool_cdus=True
)



# Option A: Specify chunks explicitly
train_loader, val_loader, test_loader, normalizer = create_tno_data_loaders(
    data_path=DATA_PATH,
    tno_config=tno_seq_config,
    batch_size=BATCH_SIZE,
    train_chunks=[0,1,2,3],
    val_chunks=[4],
    test_chunks=[5],
    num_workers=NUM_WORKERS
    
)


# Inspect sample batch
print("\nSample batch shapes:")
sample_batch = get_sample_batch(train_loader)
print_batch_shapes(sample_batch)



# =========================================================================
# Step 2: Create Model
# =========================================================================
print("\n" + "="*60)
print("Step 2: Creating TNO Model")
print("="*60)

model = create_tno_model(
    input_dim=3,              # [Q_flow, T_Air, T_ext]
    output_dim=12,            # 12 CDU output variables
    history_length=HISTORY_LENGTH,
    prediction_horizon=PREDICTION_HORIZON,
    latent_dim=64,
    d_model=32,
    branch_type='conv1d',     # Options: 'conv1d', 'lstm', 'transformer', 'mlp'
    tbranch_type='conv1d',
    dropout=0.1
)

# Print model info
param_counts = model.get_num_parameters()
print(f"\nModel Parameters:")
for name, count in param_counts.items():
    print(f"  {name}: {count:,}")

# =========================================================================
# Step 3: Create Denormalization Function
# =========================================================================
print("\n" + "="*60)
print("Step 3: Setting up Denormalization")
print("="*60)

def denormalize(y_pred: torch.Tensor) -> torch.Tensor:
    """Denormalize predictions for physics-informed loss computation."""
    pred_device = y_pred.device
    pred_dtype = y_pred.dtype
    
    # Use first 12 output stats (per-CDU outputs)
    mean = torch.tensor(normalizer.mean_out[:12], device=pred_device, dtype=pred_dtype)
    std = torch.tensor(normalizer.std_out[:12], device=pred_device, dtype=pred_dtype)
    
    return y_pred * std + mean

print("Denormalization function created")




# =========================================================================
# Step 4: Create Trainer
# =========================================================================
print("\n" + "="*60)
print("Step 4: Creating Trainer")
print("="*60)

trainer = create_tno_trainer(
    model=model,
    learning_rate=LEARNING_RATE,
    weight_decay=1e-5,
    lambda_data=1.0,
    lambda_physics=0.1,
    lambda_temp_order=0.5,
    lambda_positivity=0.1,
    lambda_smoothness=0.05,
    lambda_energy=0.1,
    num_epochs=NUM_EPOCHS,
    checkpoint_dir=CHECKPOINT_DIR,
    denormalize_fn=denormalize,
    device=device
) 
print(f"Trainer created, checkpoints will be saved to: {CHECKPOINT_DIR}")


# =========================================================================
# Step 5: Train Model
# =========================================================================
print("\n" + "="*60)
print("Step 5: Training Model")
print("="*60)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE
)




# =========================================================================
# Step 6: Evaluate on Test Set
# =========================================================================
print("\n" + "="*60)
print("Step 6: Evaluating on Test Set")
print("="*60)

test_losses = trainer.validate(test_loader)

print("\nTest Results:")
for key, value in test_losses.items():
    print(f"  {key}: {value:.6f}")

# =========================================================================
# Step 7: Make Predictions
# =========================================================================
print("\n" + "="*60)
print("Step 7: Making Sample Predictions")
print("="*60)

model.eval()
with torch.no_grad():
    # Get a batch
    batch = next(iter(test_loader))
    u_hist = batch['u_hist'].to(device)
    y_hist = batch['y_hist'].to(device)
    y_future = batch['y_future'].to(device)
    
    # Predict
    y_pred = model(u_hist, y_hist)
    
    # Denormalize for comparison
    y_pred_denorm = denormalize(y_pred)
    y_true_denorm = denormalize(y_future)
    
    # Calculate metrics
    mse = torch.mean((y_pred - y_future) ** 2).item()
    mae = torch.mean(torch.abs(y_pred - y_future)).item()
    
    print(f"\nPrediction shape: {y_pred.shape}")
    print(f"MSE (normalized): {mse:.6f}")
    print(f"MAE (normalized): {mae:.6f}")
    
    # Per-variable MSE
    print("\nPer-variable MSE (denormalized):")
    var_names = [
        'V_flow_prim', 'V_flow_sec', 'W_CDUP',
        'T_prim_s', 'T_prim_r', 'T_sec_s', 'T_sec_r',
        'p_prim_s', 'p_prim_r', 'p_sec_s', 'p_sec_r', 'htc'
    ]
    for i, name in enumerate(var_names):
        var_mse = torch.mean((y_pred_denorm[..., i] - y_true_denorm[..., i]) ** 2).item()
        print(f"  {name}: {var_mse:.4f}")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Best validation loss: {trainer.best_val_loss:.6f}")
print(f"Checkpoints saved to: {CHECKPOINT_DIR}")




