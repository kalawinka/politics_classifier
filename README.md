# Detecting scientific articles from the political science field using scientific abstracts

This tutorial introduces the process of detecting scientific articles in the political science domain using machine learning models. It provides a step-by-step guide to implementing these techniques with practical examples.

## Learning Objectives

By the end of this tutorial, you will be able to
- detect scientific articles from the political science domain using pretrained models and the transformers library

## Social Science Use Cases

The methods described in this tutorial can be applied to various social science use cases, such as:

- Analyzing political discourse
- Categorizing research papers in political science
- Identifying trends in political research topics

## Target Audience (Difficulty level)
- This tutorial is aimed at beginners with some knowledge in Python.

## Prerequisites
- Prior knowledge of Python programming is required for this tutorial
- Prior knowledge of Jupyter/IPython Notebook usage is required for this tutorial
- Prior knowledge of Google Colab usage is required for this tutorial

## What Length - Duration

The tutorial is designed to be completed in approximately 1 hour, depending on your familiarity with the prerequisites.

## Environment Setup

 - In order to run this tutorial, you need at least Python >= 3.11.4  
 - The following dependencies should also install the required packages

```bash
# Install packages for Jupiter Notebook environment
!pip install -r binder/requirements.txt
```

## Technical Details 

We first load the tokenizer and model from Huggingface. 
**Hugging Face is an open-source platform providing state-of-the-art machine learning models and tools for natural language processing tasks.**

```bash
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('kalawinka/SSciBERT_politics')
# load classification model
model = AutoModelForSequenceClassification.from_pretrained('kalawinka/SSciBERT_politics')
# initiaize classification pipeline
# optional argument device: "cuda" or "cpu"
pipe = pipeline("text-classification", model=model, tokenizer = tokenizer, max_length=512, truncation=True)
```

To run the model we input a sample text of pipeline 
```bash

# use classification pipline with a single text
example = '''Germaparl is a collection of protocols of 72 years of parliamentary debates in the German Bundestag. Analysis of parliamentary debates can reveal the hidden programmatic-ideological positions of political parties. From the linguistic perspective, parliamentary debates can be analysed in terms of political discourse, sentiment and position-taking, and cross-cultural and gender differences. Our main focus is the analysis of interjections or ’Zwischenrufe’ and calls to order or ’Ordnungsrufe’. Calls to order are a good tool to analyse the negativity in politics. Moreover, we believe that both calls to order and interjections are able to reveal tendencies in the country’s political scene, collaboration patterns between political parties and the impact and productivity of a single politician or political party. To our best knowledge calls to order and interjections in the Germaparl corpus were not analysed by other researchers.'''

out = pipe(example)

print(out)

# this outputs ----
# [{'label': 'politics', 'score': 0.9995762705802917}]

```

You can similarly follow this for huge corpus of data. Please read the [notebook](Politics-Classifier.ipynb).

## Takeaways and Conclusion

By the end of this tutorial, you will have learned:

- How to use pretrained models to classify scientific texts
- Practical applications of machine learning in the political science domain
- Insights into leveraging transformers for text classification tasks
- Lets dive further 

## References

- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers/
- Python Official Documentation: https://docs.python.org/3/

## Contact Details
- Email: nina.smirnova@gesis.org
Huggingface: https://huggingface.co/kalawinka \
Research intersets: NLP, Machine Learning, Computational Linguistics, LLMs, Text Minings