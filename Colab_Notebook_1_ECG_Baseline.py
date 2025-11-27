# ============================================================================
# GOOGLE COLAB NOTEBOOK 1: ECG Baseline Model for MI Prediction
# ============================================================================
# Copy this entire file into a Google Colab notebook and run cell by cell
# Expected runtime: 2-3 hours on Colab free GPU
# Expected AUROC: 0.82-0.85
# ============================================================================

# ============================================================================
# CELL 1: Setup and Installation
# ============================================================================
"""
Run this cell first to install all required packages.
This will take ~2-3 minutes.
"""

!pip install -q wfdb pandas numpy scikit-learn matplotlib seaborn
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q tqdm

print("✅ All packages installed successfully!")

# ============================================================================
# CELL 2: Download PTB-XL Dataset
# ============================================================================
"""
Download the PTB-XL ECG dataset (21,837 ECGs, ~2GB).
This will take ~5-10 minutes depending on connection speed.
"""

import os
import urllib.request
import zipfile

# Download dataset
print("📥 Downloading PTB-XL dataset...")
url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
zip_path = "ptb-xl.zip"

if not os.path.exists("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"):
    urllib.request.urlretrieve(url, zip_path)
    print("📦 Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    os.remove(zip_path)
    print("✅ Dataset downloaded and extracted!")
else:
    print("✅ Dataset already exists!")

# ============================================================================
# CELL 3: Load and Explore Data
# ============================================================================
"""
Load the dataset and explore its structure.
"""

import pandas as pd
import numpy as np
import wfdb
import ast

# Set path
DATA_PATH = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'

# Load database
Y = pd.read_csv(DATA_PATH + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

print(f"📊 Total ECG records: {len(Y)}")
print(f"\n📋 Columns: {Y.columns.tolist()}")
print(f"\n🔍 Sample data:")
print(Y.head())

# ============================================================================
# CELL 4: Extract MI Cases
# ============================================================================
"""
Identify myocardial infarction (MI) cases from diagnostic codes.
"""

# MI-related diagnostic codes in PTB-XL
MI_CODES = ['IMI', 'AMI', 'LMI', 'PMI', 'ASMI', 'ILMI', 'ALMI', 
            'INJAS', 'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA']

def has_mi(scp_codes_dict):
    """Check if any MI code is present"""
    for code in MI_CODES:
        if code in scp_codes_dict:
            return 1
    return 0

# Create MI label
Y['mi_label'] = Y.scp_codes.apply(has_mi)

# Statistics
mi_count = Y.mi_label.sum()
non_mi_count = (1 - Y.mi_label).sum()

print(f"✅ MI cases: {mi_count} ({mi_count/len(Y)*100:.1f}%)")
print(f"✅ Non-MI cases: {non_mi_count} ({non_mi_count/len(Y)*100:.1f}%)")
print(f"\n📊 Class imbalance ratio: 1:{non_mi_count/mi_count:.1f}")

# ============================================================================
# CELL 5: Load ECG Signals Function
# ============================================================================
"""
Function to load ECG waveforms from files.
"""

def load_ecg_signal(ecg_id, sampling_rate=100):
    """
    Load ECG signal for given ID
    
    Args:
        ecg_id: ECG identifier
        sampling_rate: 100 or 500 Hz
        
    Returns:
        signal: numpy array of shape (5000, 12) for 100Hz
    """
    if sampling_rate == 100:
        data = wfdb.rdsamp(DATA_PATH + f'records100/{ecg_id}')
    else:
        data = wfdb.rdsamp(DATA_PATH + f'records500/{ecg_id}')
    
    signal = data[0]  # Get signal data
    return signal

# Test loading one ECG
sample_id = Y.index[0]
sample_ecg = load_ecg_signal(sample_id)
print(f"✅ Sample ECG shape: {sample_ecg.shape}")
print(f"   (Time points: {sample_ecg.shape[0]}, Leads: {sample_ecg.shape[1]})")

# ============================================================================
# CELL 6: Visualize Sample ECGs
# ============================================================================
"""
Visualize MI vs Non-MI ECG examples.
"""

import matplotlib.pyplot as plt

# Get one MI and one non-MI example
mi_example = Y[Y.mi_label == 1].index[0]
non_mi_example = Y[Y.mi_label == 0].index[0]

mi_ecg = load_ecg_signal(mi_example)
non_mi_ecg = load_ecg_signal(non_mi_example)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(15, 8))

# MI ECG
axes[0].plot(mi_ecg[:, 0])  # Lead I
axes[0].set_title('MI Case - Lead I', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Amplitude (mV)')
axes[0].grid(True, alpha=0.3)

# Non-MI ECG
axes[1].plot(non_mi_ecg[:, 0])  # Lead I
axes[1].set_title('Non-MI Case - Lead I', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time (samples)')
axes[1].set_ylabel('Amplitude (mV)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ ECG visualization complete!")

# ============================================================================
# CELL 7: Create Train/Val/Test Splits
# ============================================================================
"""
Split data into training, validation, and test sets.
"""

from sklearn.model_selection import train_test_split

# Get stratified splits
train_val, test = train_test_split(
    Y, test_size=0.15, random_state=42, stratify=Y.mi_label
)

train, val = train_test_split(
    train_val, test_size=0.176, random_state=42, stratify=train_val.mi_label  # 0.176 * 0.85 ≈ 0.15
)

print(f"📊 Dataset splits:")
print(f"   Train: {len(train)} ({len(train)/len(Y)*100:.1f}%)")
print(f"   Val:   {len(val)} ({len(val)/len(Y)*100:.1f}%)")
print(f"   Test:  {len(test)} ({len(test)/len(Y)*100:.1f}%)")

print(f"\n✅ MI distribution:")
print(f"   Train: {train.mi_label.sum()} / {len(train)} = {train.mi_label.mean()*100:.1f}%")
print(f"   Val:   {val.mi_label.sum()} / {len(val)} = {val.mi_label.mean()*100:.1f}%")
print(f"   Test:  {test.mi_label.sum()} / {len(test)} = {test.mi_label.mean()*100:.1f}%")

# ============================================================================
# CELL 8: PyTorch Dataset Class
# ============================================================================
"""
Create PyTorch Dataset for ECG data.
"""

import torch
from torch.utils.data import Dataset, DataLoader

class PTBXLDataset(Dataset):
    def __init__(self, dataframe, data_path, sampling_rate=100):
        self.df = dataframe.reset_index()
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_id = row['ecg_id']
        label = row['mi_label']
        
        # Load ECG signal
        if self.sampling_rate == 100:
            data = wfdb.rdsamp(self.data_path + f'records100/{ecg_id}')
        else:
            data = wfdb.rdsamp(self.data_path + f'records500/{ecg_id}')
        
        signal = data[0]  # Shape: (5000, 12)
        
        # Transpose to (12, 5000) for Conv1d
        signal = torch.FloatTensor(signal.T)
        label = torch.FloatTensor([label])
        
        return signal, label

# Create datasets
train_dataset = PTBXLDataset(train, DATA_PATH)
val_dataset = PTBXLDataset(val, DATA_PATH)
test_dataset = PTBXLDataset(test, DATA_PATH)

# Create dataloaders
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ Dataloaders created!")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")

# Test loading one batch
sample_batch = next(iter(train_loader))
print(f"\n✅ Sample batch shapes:")
print(f"   ECG: {sample_batch[0].shape}")  # (batch, 12, 5000)
print(f"   Label: {sample_batch[1].shape}")  # (batch, 1)

# ============================================================================
# CELL 9: Define CNN Model
# ============================================================================
"""
Define a ResNet-inspired CNN for ECG classification.
"""

import torch.nn as nn
import torch.nn.functional as F

class ECGResNet(nn.Module):
    def __init__(self, num_leads=12):
        super(ECGResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Block 1
        identity = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if identity.shape[1] != x.shape[1]:
            identity = nn.Conv1d(identity.shape[1], x.shape[1], kernel_size=1).to(x.device)(identity)
        x = F.relu(x + identity)
        x = F.max_pool1d(x, 2)
        
        # Block 2
        identity = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        if identity.shape[1] != x.shape[1]:
            identity = nn.Conv1d(identity.shape[1], x.shape[1], kernel_size=1).to(x.device)(identity)
        x = F.relu(x + identity)
        x = F.max_pool1d(x, 2)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ECGResNet().to(device)

print(f"✅ Model created on device: {device}")
print(f"\n📊 Model architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Parameters:")
print(f"   Total: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

# ============================================================================
# CELL 10: Training Setup
# ============================================================================
"""
Setup loss function, optimizer, and training parameters.
"""

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Loss function (weighted for class imbalance)
pos_weight = torch.tensor([non_mi_count / mi_count]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# Training parameters
NUM_EPOCHS = 30
BEST_VAL_AUROC = 0.0

print(f"✅ Training setup complete!")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Learning rate: 0.001")
print(f"   Pos weight: {pos_weight.item():.2f}")

# ============================================================================
# CELL 11: Training and Validation Functions
# ============================================================================
"""
Define training and validation functions.
"""

from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for ecg, labels in pbar:
        ecg, labels = ecg.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(ecg)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(loader)
    auroc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auroc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for ecg, labels in tqdm(loader, desc='Validation'):
            ecg, labels = ecg.to(device), labels.to(device)
            
            outputs = model(ecg)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    auroc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auroc

print("✅ Training functions defined!")

# ============================================================================
# CELL 12: Train the Model
# ============================================================================
"""
Train the model for NUM_EPOCHS epochs.
This will take ~2-3 hours on Colab free GPU.
"""

import time

# Training history
history = {
    'train_loss': [],
    'train_auroc': [],
    'val_loss': [],
    'val_auroc': []
}

print("🚀 Starting training...")
print("=" * 60)

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 60)
    
    # Train
    train_loss, train_auroc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_auroc = validate(model, val_loader, criterion, device)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_auroc'].append(train_auroc)
    history['val_loss'].append(val_loss)
    history['val_auroc'].append(val_auroc)
    
    # Print metrics
    print(f"\n📊 Results:")
    print(f"   Train Loss: {train_loss:.4f} | Train AUROC: {train_auroc:.4f}")
    print(f"   Val Loss:   {val_loss:.4f} | Val AUROC:   {val_auroc:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_auroc)
    
    # Save best model
    if val_auroc > BEST_VAL_AUROC:
        BEST_VAL_AUROC = val_auroc
        torch.save(model.state_dict(), 'best_ecg_model.pth')
        print(f"   ✅ New best model saved! (AUROC: {val_auroc:.4f})")

elapsed_time = time.time() - start_time
print(f"\n✅ Training complete!")
print(f"   Total time: {elapsed_time/60:.1f} minutes")
print(f"   Best Val AUROC: {BEST_VAL_AUROC:.4f}")

# ============================================================================
# CELL 13: Plot Training History
# ============================================================================
"""
Visualize training progress.
"""

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# AUROC plot
ax2.plot(history['train_auroc'], label='Train AUROC', linewidth=2)
ax2.plot(history['val_auroc'], label='Val AUROC', linewidth=2)
ax2.axhline(y=BEST_VAL_AUROC, color='r', linestyle='--', label=f'Best Val AUROC: {BEST_VAL_AUROC:.4f}')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('AUROC', fontsize=12)
ax2.set_title('Training and Validation AUROC', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Training history plotted!")

# ============================================================================
# CELL 14: Evaluate on Test Set
# ============================================================================
"""
Evaluate the best model on the test set.
"""

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix

# Load best model
model.load_state_dict(torch.load('best_ecg_model.pth'))
model.eval()

# Get predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for ecg, labels in tqdm(test_loader, desc='Testing'):
        ecg = ecg.to(device)
        outputs = model(ecg)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()

# Calculate metrics
test_auroc = roc_auc_score(all_labels, all_preds)
precision, recall, _ = precision_recall_curve(all_labels, all_preds)
test_auprc = auc(recall, precision)

# Binary predictions (threshold = 0.5)
binary_preds = (all_preds > 0.5).astype(int)
accuracy = accuracy_score(all_labels, binary_preds)

print("=" * 60)
print("📊 TEST SET RESULTS")
print("=" * 60)
print(f"✅ AUROC: {test_auroc:.4f}")
print(f"✅ AUPRC: {test_auprc:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")
print("\n📋 Classification Report:")
print(classification_report(all_labels, binary_preds, target_names=['Non-MI', 'MI']))

# Confusion matrix
cm = confusion_matrix(all_labels, binary_preds)
print("\n📊 Confusion Matrix:")
print(f"                Predicted")
print(f"              Non-MI    MI")
print(f"Actual Non-MI  {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"       MI      {cm[1,0]:5d}  {cm[1,1]:5d}")

# ============================================================================
# CELL 15: Plot ROC and PR Curves
# ============================================================================
"""
Visualize ROC and Precision-Recall curves.
"""

from sklearn.metrics import roc_curve

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ROC Curve
ax1.plot(fpr, tpr, linewidth=3, label=f'ECG Baseline (AUROC = {test_auroc:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Precision-Recall Curve
ax2.plot(recall, precision, linewidth=3, label=f'ECG Baseline (AUPRC = {test_auprc:.3f})')
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ ROC and PR curves saved as 'roc_pr_curves.png'")

# ============================================================================
# CELL 16: Save Results
# ============================================================================
"""
Save all results for the paper.
"""

# Save results to file
results = {
    'test_auroc': test_auroc,
    'test_auprc': test_auprc,
    'test_accuracy': accuracy,
    'best_val_auroc': BEST_VAL_AUROC,
    'training_time_minutes': elapsed_time / 60,
    'num_epochs': NUM_EPOCHS,
    'total_params': total_params
}

import json
with open('ecg_baseline_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("✅ Results saved to 'ecg_baseline_results.json'")
print("\n📊 FINAL SUMMARY:")
print("=" * 60)
print(f"Model: ECG Baseline (ResNet-inspired CNN)")
print(f"Dataset: PTB-XL (21,837 ECGs)")
print(f"Test AUROC: {test_auroc:.4f}")
print(f"Test AUPRC: {test_auprc:.4f}")
print(f"Training time: {elapsed_time/60:.1f} minutes")
print(f"Parameters: {total_params:,}")
print("=" * 60)
print("\n🎉 EXPERIMENT COMPLETE!")
print("\n📝 Next steps:")
print("1. Use this AUROC ({:.4f}) as your baseline in the paper".format(test_auroc))
print("2. Run Notebook 2 to implement multi-agent system")
print("3. Compare results and create tables for the paper")

# Download files (uncomment to download)
# from google.colab import files
# files.download('best_ecg_model.pth')
# files.download('ecg_baseline_results.json')
# files.download('roc_pr_curves.png')
