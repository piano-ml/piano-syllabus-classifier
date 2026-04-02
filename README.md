# Piano Syllabus Classifier

Transformer-based classifier that predicts the piano difficulty grade (Initial → Grade 10) from MIDI files, using [MidiTok](https://github.com/Natooz/MidiTok) REMI tokenization and a custom Transformer encoder trained with Hugging Face Trainer.

## Requirements

### python deps

```bash
pip install miditok transformers[torch] torch evaluate scikit-learn accelerate seaborn safetensors
```

### dataset deps (see credits)

From: https://zenodo.org/records/14794592
Get data.json from new_clean_data.json 
Get mid.zip and unzip it to get a ./mid/ directory


## Project structure

| File | Description |
|---|---|
| `train_ps_classifier.py` | Main CLI entry point — runs training + test evaluation end-to-end |
| `common.py` | Shared config, label mapping, tokenizer builder, utilities |
| `checks.py` | Data validation, class distribution analysis and plots |
| `model.py` | `MidiClassifier` — Transformer encoder + classification head |
| `training.py` | Dataset creation, collator, HF Trainer setup, training loop |
| `evaluate_model.py` | Test-set evaluation, confusion matrix, per-class report |
| `inference.py` | Predict grade for a new MIDI file |
| `data.json` | Label file mapping piece keys → metadata with `ps` field |
| `mid/` | Directory containing ~7900 piano MIDI files |

## Train

```bash
python train_ps_classifier.py \
    --midi_dir mid \
    --labels_json data.json \
    --output_dir ./ps_model \
    --epochs 6
```

All arguments with defaults:

```
--midi_dir mid              # MIDI files directory
--labels_json data.json     # Label JSON file
--output_dir ./ps_model  # Output directory for model & plots
--epochs 6                  # Training epochs
--batch_size 16             # Batch size
--lr 3e-4                   # Learning rate
--max_seq_len 1024          # Max token sequence length
--d_model 256               # Transformer embedding dimension
--nhead 8                   # Attention heads
--num_layers 4              # Transformer encoder layers
--dim_feedforward 512       # FFN dimension
--dropout 0.1               # Dropout rate
--seed 42                   # Random seed
--pre_tokenize              # Pre-tokenize all files (faster, more RAM)
```

## Inference

Predict grade for a single MIDI file:

```bash
python inference.py \
    --model_dir ./ps_model \
    --midi_file path/to/piece.mid
```

## Evaluate a saved model

```bash
python evaluate_model.py \
    --model_dir ./ps_model \
    --midi_dir mid \
    --labels_json data.json \
    --output_dir ./eval_results
```

## Output

Training produces these files in `--output_dir`:

- `best_model/` — saved model weights
- `tokenizer.json` — REMI tokenizer config
- `class_distribution.png` — dataset label distribution
- `split_distribution.png` — train/val/test distribution
- `training_curves.png` — loss, accuracy, F1 curves
- `confusion_matrix.png` — test confusion matrix
- `confusion_matrix_normalized.png` — normalized confusion matrix
- `per_class_accuracy.png` — per-grade accuracy bar chart
- `prediction_distribution.png` — true vs predicted distributions
- `test_report.txt` — classification report

## Credits

Ramoneda, P., Lee, M., Jeong, D., Valero-Mas, J. J., & Serra, X. (2025). Piano Syllabus Dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14794592

