# In-Context Learning (ICL) with DeepSeek Model

## Overview

This notebook demonstrates in-context learning (ICL) using the **DeepSeek-R1:14B** model for text summarization. It leverages the **CNN/DailyMail dataset** to evaluate summarization performance. The model is accessed via an API, which is assumed to be running locally.

## Requirements

### Dependencies

Ensure you have the following Python packages installed:

```bash
pip install pandas datasets openai evaluate requests
```

### API Setup

The DeepSeek model is accessed through an API. Before running the notebook, ensure:

1. The API server is running locally at `http://localhost:11434/v1`
2. You have the necessary API key (set as `OLLAMA_API_KEY` in the script)

## Execution Steps

### 1. Load Required Libraries

The script imports necessary modules for dataset handling, API communication, and evaluation.

### 2. Load Dataset

The CNN/DailyMail dataset is loaded using the `datasets` library:

```python
from datasets import load_dataset
cnn_dailymail_data = load_dataset('abisee/cnn_dailymail', '1.0.0')
```

### 3. Define API Communication

The script uses OpenAI's API client to interact with the locally hosted DeepSeek model:

```python
from openai import OpenAI
openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
```

Ensure your local API server is running before executing this step.

### 4. Prepare Message for In-Context Learning

The function `prepareMessage` constructs the prompt for in-context learning. It:

- Selects random examples from the dataset as in-context examples.
- Formats them as article-summary pairs.
- Enforces constraints such as limiting the summary length to 50 words.
- Ensures no markdown is included in the summary.

#### Function Definition:

```python
def prepareMessage(contextSampleSize=0, testArticleIndex=1):
    randomVal = generateRandomIndex(0, trainDSetSize)
    message = f"""
Article: {icl_examples[randomVal]["article"]}\nSummary: {icl_examples[randomVal]["summary"]}\n\n"""
    for i in range(0, contextSampleSize):
        randomVal = generateRandomIndex(0, trainDSetSize)
        message += f"Article: {icl_examples[randomVal]["article"]}\nSummary: {icl_examples[randomVal]["summary"]}\n\n"
    message += """
The above contents are Article-Summary pairs for in-context learning. The last article below doesn't have a summary. Please provide a concise summary (≤50 words) in plain text.
"""
    message += f"Article: {first300ArticleTest[testArticleIndex]}\nSummary: ";
    return message
```

### 5. Generate Summaries

The script sends text inputs to the DeepSeek model and retrieves the generated summaries. It randomly selects test examples from the dataset and evaluates the model’s performance.

### 6. Evaluate Performance

ROUGE scores are used to assess the quality of generated summaries:

```python
from evaluate import load
rouge = load('rouge')
scores = rouge.compute(predictions=generated_summaries, references=reference_summaries)
```

## Expected Output

- Generated summaries for CNN/DailyMail articles.
- ROUGE scores indicating the quality of summarization.

## Troubleshooting

- **API Connection Error:** Ensure your API server is running at the specified base URL.
- **Missing Dependencies:** Install required Python packages using `pip install`.
- **Dataset Loading Issues:** Ensure internet access is available to fetch the dataset.

By following these steps, your supervisor should be able to execute the notebook and reproduce the results effectively.
