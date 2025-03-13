## Overview
This project implements a Spam Classification Model using a custom-built GPT-like architecture. 
The model has been developed from scratch, incorporating various components of a language model (LLM) before integrating weights from OpenAI's GPT-2. 
This README provides an overview of the project's structure, setup instructions, and usage guidelines.

## Features:
Custom Implementation - All components of the model, including the tokenizer, attention mechanism, and dataset handling, are implemented from scratch.<br>
Spam Classification - The model is trained to classify text messages as either "spam" or "ham" (not spam).<br>
Data Handling - The project includes functionality to download and preprocess the SMS Spam Collection dataset.<br>
Visualization - Training and validation losses are plotted for analysis.<br>

## Requirements:
Python 3.7 or higher
PyTorch
TensorFlow
NumPy
Pandas
Matplotlib
tqdm
tiktoken

## Dataset
The project uses the SMS Spam Collection dataset, which can be downloaded automatically. 
The dataset consists of a collection of SMS messages labeled as "spam" or "ham".

## Downloading the Dataset
The dataset is downloaded and extracted using the following function:
```python
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
```

## Model Architecture
The model is based on the GPT architecture and includes:
![Total number of parameters](https://github.com/user-attachments/assets/b19af62b-82c1-434e-9afe-a2cd107f0c04)

Tokenization: Uses the tiktoken library for encoding text.
Embedding Layers: Token and position embeddings are implemented.<br>
Transformer Blocks: Multiple transformer blocks for attention mechanisms.<br>
Feed Forward Neural Networks: For processing the outputs of the attention layers.<br>
Output Layer: A neural network on top of the output layer classifies the input text as spam or not spam by analyzing 
the final output representations (logits) from the transformer blocks.<br>

## Training the Model
The model is trained using the following function:
```python
train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter)
```

## Parameters
model: The GPT model instance.<br>
train_loader: DataLoader for training data.<br>
val_loader: DataLoader for validation data.<br>
optimizer: Optimizer for model training.<br>
device: Device to run the model (CPU or GPU).<br>
num_epochs: Number of training epochs.<br>
eval_freq: Frequency of evaluation during training.<br>
eval_iter: Number of iterations for evaluation.<br>

## Evaluation
The model's performance can be evaluated using the following function:
```python
calc_accuracy_loader(data_loader, model, device, num_batches=None)
```

## Inference
To classify a new text message, use the classify_review function:
```python
classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256)
```

## Visualization
Training and validation losses are plotted using Matplotlib:
```python
plot_values(epochs_seen, examples_seen, train_losses, val_losses)
```
![WhatsApp Image 2025-03-10 at 21 32 36](https://github.com/user-attachments/assets/b4637b38-7c0f-4241-b388-8f4a25e85741)


## Conclusion
This project implements a spam classification model using a custom language model architecture. 
It focuses on understanding transformer mechanisms and lays the groundwork for further exploration in 
natural language processing, such as top-k sampling, temperature scaling, and multimodality.

## Reference
[Build a Large Language Model (From Scratch)]([https://sebastianraschka.com/books/](https://www.manning.com/books/build-a-large-language-model-from-scratch?utm_source=raschka&utm_medium=affiliate&utm_campaign=book_raschka_build_12_12_23&a_aid=raschka&a_bid=4c2437a0&chan=mm_website))
