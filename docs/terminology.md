# NLP & Deep Learning Terminology (Beginner Glossary)

Short, plain‑English definitions for the terms used throughout this repo. Wording is paraphrased and tailored for beginners.

## Core ideas

- Text representation: Turning raw words into numbers so a computer can work with them.
- Token: A cleaned, lowercased word unit used in our examples.
- Vocabulary: The set of unique tokens found in a text or corpus (often sorted for consistency).

## Classic representations

- One‑hot encoding: A simple vector for a token with a single 1 at its position in the vocabulary and 0s elsewhere. No meaning beyond identity.
- Bag of Words (BoW): Counts how often each word appears in a document, ignoring order.
- Term Frequency (TF): The frequency of a word within one document (often word count divided by document length).
- Inverse Document Frequency (IDF): A weight that lowers the importance of words that appear in many documents and raises those that are rare.
- TF‑IDF: The product of TF and IDF, highlighting words that are frequent in one document but uncommon overall.

## Learned representations

- Word embeddings: Dense vectors learned from data that place related words near each other in a continuous space (e.g., word2vec). Capture similarity beyond exact matches.

## Neural sequence models

- Recurrent Neural Network (RNN): A neural net that processes sequences one step at a time, carrying forward a hidden state.
- Long Short‑Term Memory (LSTM): A type of RNN designed to better remember information by using gates to control what to keep or forget.
- Gated Recurrent Unit (GRU): A streamlined RNN similar to LSTM that uses gates to manage memory with fewer parameters.
- Convolutional Neural Network (CNN) for text: A model that scans over word windows with filters to pick up local patterns like phrases or n‑grams.

## Datasets and pipelines

- Corpus: A collection of texts used for building vocabularies, statistics, and models.
- Preprocessing: Steps like lowercasing, removing punctuation, and tokenizing to make text consistent.
- Feature matrix: A table where rows are documents or tokens and columns are features (e.g., words); the entries are counts, weights, or 0/1 indicators.

## Modern context

- Language model (LM): A model that predicts or scores sequences of words; modern large language models generate or understand text.
- Large Language Model (LLM): A very large LM trained on extensive text data to perform many language tasks (e.g., answering questions, summarizing).

## Quick contrasts

- One‑hot vs. embeddings: One‑hot only marks identity; embeddings encode similarity and meaning.
- BoW vs. TF‑IDF: Both ignore order; TF‑IDF downweights common words and highlights distinctive ones.

## Tips

- When reading tables: rows usually mean samples (documents or tokens), columns are features (vocabulary). Column order often follows the sorted vocabulary.
- Start small: Validate pipelines with tiny examples before scaling up.
