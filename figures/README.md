# Figures for MICCAI 2026 Paper

This directory contains all figures for the paper.

## Figure List

### Figure 1: Multi-Agent Architecture
- **File**: `figure1_architecture.png` / `.pdf`
- **Description**: System architecture showing ECG Agent, Clinical Agent, and Synthesis Agent
- **Use in paper**: Section 3 (Methods), Figure 1

### Figure 2: ROC Curves
- **File**: `figure2_roc_curves.png` / `.pdf`
- **Description**: ROC curves comparing baseline methods vs. multi-agent system
- **Use in paper**: Section 4 (Experiments), Figure 2

### Figure 3: Ablation Study
- **File**: `figure3_ablation.png` / `.pdf`
- **Description**: Bar chart showing performance drop when removing components
- **Use in paper**: Section 4 (Experiments), Figure 3

### Figure 4: Training Curves
- **File**: `figure4_training_curves.png` / `.pdf`
- **Description**: Training and validation loss/AUROC over epochs
- **Use in paper**: Section 4 (Experiments), Figure 4

### Figure 5: Performance Table
- **File**: `figure5_performance_table.png` / `.pdf`
- **Description**: Comparison table of all methods with AUROC, AUPRC, Sensitivity
- **Use in paper**: Section 4 (Experiments), Table 1

## How to Use

### In LaTeX (Overleaf)

1. Upload all figures to Overleaf `figures/` folder
2. Reference in your paper:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\textwidth]{figures/figure1_architecture.pdf}
\caption{Multi-agent system architecture...}
\label{fig:architecture}
\end{figure}
```

### Regenerate Figures

If you need to update figures with new results:

```bash
python generate_figures.py
```

## Figure Specifications

- **Format**: PNG (300 DPI) and PDF (vector)
- **Size**: Optimized for MICCAI 8-page limit
- **Style**: Publication-quality, color-blind friendly
- **Fonts**: Sans-serif, readable when printed

## Notes

- All figures are generated programmatically
- Edit `generate_figures.py` to update values
- PDF versions recommended for LaTeX (better quality)
- PNG versions for presentations/slides
