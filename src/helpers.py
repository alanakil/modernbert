# %%
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# %%
def compute_metrics(logits, labels, label_names):
    """Compute classification metrics from raw logits and integer labels.

    Args:
        logits: numpy array of shape (n_samples, n_classes)
        labels: numpy array of shape (n_samples,) with integer class indices
        label_names: list of class name strings, length n_classes

    Returns:
        dict with keys: accuracy, macro_precision, macro_recall, macro_f1,
        auc_roc, and f1_class_0 … f1_class_{n-1}
    """
    predicted = np.argmax(logits, axis=1)
    probs = softmax(logits, axis=1)

    accuracy = accuracy_score(labels, predicted)
    macro_precision = precision_score(labels, predicted, average="macro", zero_division=0)
    macro_recall = recall_score(labels, predicted, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, predicted, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels, predicted, average=None, zero_division=0)

    try:
        auc_roc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc_roc = float("nan")

    result = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "auc_roc": auc_roc,
    }
    for i, f1 in enumerate(per_class_f1):
        result[f"f1_class_{i}"] = f1

    return result


# %%
def identify_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


# %%
def tokenize_newsgroups(example, tokenizer):
    # Tokenize the 'text' field and apply truncation.
    return tokenizer(example['text'], truncation=True)


# %%
# Define a function to compute the token length for each example
def compute_length(example, tokenizer):
    # Tokenize the 'text' field without truncation
    tokens = tokenizer.tokenize(example['text'])
    # Store the token count in a new field 'length'
    example['length'] = len(tokens)
    return example


# %%
def tokenize_and_trim(example, tokenizer, max_length=200):
    # Tokenize the 'text' field with truncation enabled and a specified max_length.
    # The returned dictionary will include fields like 'input_ids' and 'attention_mask'.
    return tokenizer(example['text'], truncation=True, max_length=max_length)


# %%