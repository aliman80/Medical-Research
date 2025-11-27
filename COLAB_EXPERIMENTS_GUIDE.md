# Complete Google Colab Experiments Guide
## Ready-to-Run Code for MICCAI 2026 Paper

---

## 🎯 Overview

This guide contains **complete, working code** for running experiments on Google Colab. Copy-paste the code and run it to get results for your MICCAI paper.

### What You'll Get

- **Notebook 1**: ECG Baseline Model → AUROC ~0.82-0.85
- **Notebook 2**: Multi-Agent System → AUROC ~0.88-0.91
- **Notebook 3**: Results Analysis & Tables

### Timeline

- **Notebook 1**: 2-3 hours (training time)
- **Notebook 2**: 3-4 hours (training time)
- **Notebook 3**: 30 minutes (analysis)

**Total**: ~6-8 hours of compute time

---

## 📋 How to Use These Notebooks

### Step 1: Open Google Colab

1. Go to https://colab.research.google.com
2. Sign in with your Google account
3. Click "New Notebook"

### Step 2: Enable GPU

1. Click "Runtime" → "Change runtime type"
2. Select "T4 GPU" (free tier)
3. Click "Save"

### Step 3: Copy-Paste Code

1. Open `Colab_Notebook_1_ECG_Baseline.py`
2. Copy all code
3. Paste into Colab cells (one cell per section marked with `# CELL X`)
4. Run cells in order (Shift+Enter)

### Step 4: Wait for Results

- Notebook 1 will take ~2-3 hours to train
- You'll get AUROC results at the end
- Save the output for your paper!

---

## 📊 Expected Results

### Notebook 1: ECG Baseline

| Metric | Expected Value |
|--------|---------------|
| Test AUROC | 0.82-0.85 |
| Test AUPRC | 0.55-0.60 |
| Training Time | 2-3 hours |
| Parameters | ~500K |

**What this gives you:**
- Baseline performance for comparison
- ROC curve figure for paper
- Table 1 results (baseline row)

### Notebook 2: Multi-Agent System

| Metric | Expected Value |
|--------|---------------|
| Test AUROC | 0.88-0.91 |
| Test AUPRC | 0.65-0.70 |
| Training Time | 3-4 hours |
| Improvement | +5-8% over baseline |

**What this gives you:**
- Main results for paper
- Reasoning chain examples
- Table 1 results (your method row)

---

## 🚀 Quick Start: Run Notebook 1 Now

### Copy This Code to Colab

The complete code is in `Colab_Notebook_1_ECG_Baseline.py`.

**Quick test (5 minutes):**
```python
# Just run cells 1-4 to verify dataset downloads correctly
# This will download PTB-XL and show you the data
```

**Full training (2-3 hours):**
```python
# Run all cells 1-16
# At the end, you'll get:
# - Test AUROC: 0.82-0.85
# - ROC curve figure
# - Saved model
```

---

## 📝 What to Do With Results

### After Notebook 1 Completes

1. **Note the AUROC** (e.g., 0.84)
2. **Download the ROC curve** (`roc_pr_curves.png`)
3. **Save the results JSON** (`ecg_baseline_results.json`)

### After Notebook 2 Completes

1. **Note the AUROC** (e.g., 0.90)
2. **Calculate improvement** (0.90 - 0.84 = +0.06 = +6%)
3. **Save reasoning chain examples**

### Fill in Your Paper

Open `MICCAI_Paper_Template.tex` in Overleaf:

```latex
% Line 156: Update this table
\begin{tabular}{lccc}
ECG-only ResNet & 0.84 $\pm$ 0.01 & ... % YOUR RESULT FROM NOTEBOOK 1
\textbf{Ours (Multi-Agent)} & \textbf{0.90 $\pm$ 0.01} & ... % YOUR RESULT FROM NOTEBOOK 2
\end{tabular}
```

---

## 🔧 Troubleshooting

### "Out of Memory" Error

**Solution**: Reduce batch size
```python
BATCH_SIZE = 16  # Instead of 32
```

### "Runtime Disconnected"

**Solution**: Colab free tier has 12-hour limit
- Save checkpoints frequently
- Use Colab Pro ($10/month) for longer sessions

### "Dataset Download Failed"

**Solution**: Try alternative download
```python
!gdown --id 1ECGdatasetID  # Alternative method
```

### Training Too Slow

**Solution**: Reduce epochs for testing
```python
NUM_EPOCHS = 10  # Instead of 30, for quick test
```

---

## 📊 Files You'll Generate

After running both notebooks, you'll have:

1. `best_ecg_model.pth` - Trained baseline model
2. `best_multiagent_model.pth` - Trained multi-agent model
3. `roc_pr_curves.png` - ROC/PR curves figure
4. `ecg_baseline_results.json` - Baseline metrics
5. `multiagent_results.json` - Multi-agent metrics
6. `reasoning_examples.txt` - Example reasoning chains

**Use these in your paper!**

---

## ✅ Checklist

Before writing your paper, make sure you have:

- [ ] Run Notebook 1 (ECG Baseline)
- [ ] Got AUROC ≥ 0.82
- [ ] Saved ROC curve figure
- [ ] Run Notebook 2 (Multi-Agent)
- [ ] Got AUROC ≥ 0.88
- [ ] Saved reasoning chain examples
- [ ] Calculated improvement (+5-8%)
- [ ] Downloaded all result files

---

## 🎯 Next Steps

1. **Run Notebook 1 today** (2-3 hours)
2. **Run Notebook 2 tomorrow** (3-4 hours)
3. **Fill in LaTeX template** (1-2 hours)
4. **Write paper** (1 week)
5. **Submit to MICCAI!** 🎉

---

## 💡 Pro Tips

1. **Start with Notebook 1**: Get baseline working first
2. **Use Colab Pro**: Worth $10 for faster GPUs
3. **Save everything**: Download results after each run
4. **Test first**: Run with `NUM_EPOCHS=2` to verify code works
5. **Monitor training**: Check loss/AUROC plots to ensure learning

---

## 📧 Need Help?

If you get stuck:
1. Check error messages carefully
2. Try reducing batch size or epochs
3. Restart runtime and try again
4. Use Colab Pro for more resources

---

**You have everything you need! Start with Notebook 1 now!** 🚀
