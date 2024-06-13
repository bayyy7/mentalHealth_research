<p>
    <img src="plot/diagram.png">
</p>

> Figure 1.	The architecture of mental health-related frame detection system on Indonesian-language twitter platform

# Automatic Categorization of Mental Health Frame in Indonesian X (Twitter) Text using Classification and Topic Detection Techniques

This paper aims to develop a machine learning model to detect mental health frames in Indonesian-language tweets on the X (Twitter) platform. Considering the global increase in mental health issues, including in Indonesia, recent literature reviews indicate that previous studies have been narrow in scope, overlooking many important issues. This paper addresses the problem by applying classification and topic detection methods across various mental health frames through multiple stages. First, this paper examines various mental health frames, resulting in 7 main labels: Awareness, Classification, Feelings and Problematization, Accessibility and Funding, Stigma, Service, Youth, and an additional label named Others. Second, it focuses on constructing a dataset of Indonesian tweets, totaling 29,068 data, by filtering tweets using the keywords "mental health" and "kesehatan mental". Third, this paper conducts data preprocessing and manual labeling of a random selection of 3,828 tweets, chosen due to the impracticality of labeling all data. Finally, the fourth stage involves conducting classification experiments using classical text features, non-contextual and contextual word embeddings, and performing topic detection experiments with three different algorithms. The experiments show that the BERT-based method achieved the highest accuracy, with 81% in the 'Others' vs. 'non-Others' classification, 80% in the seven main label classifications, and 92% in the seven main labels classification when using GPT-4-powered data augmentation. Topic detection experiments indicate that the Latent Dirichlet Allocation (LDA) and Latent Semantic Indexing (LSI) algorithms are more effective than the Hierarchical Dirichlet Process (HDP) in generating relevant keywords representing the characteristics of each main label

# Classification Techniques

### Algorithm
- Bag of Words (BoW)</br>
- TF-IDF </br>
- N-Grams</br>
- Fasttext (Using pretrained ID)</br>
- BPE (Using pretrained ID)</br>
- GloVe (Using pretrained Twitter Multilingual)</br>
- BERT</br>
### Pretrained of Non-Contextual Model 
- Fasttext
  Pretained available here : https://fasttext.cc/docs/en/crawl-vectors.html </br>
  This paper using Indonesian Word Vector (300 Dimension)</br>
- BPE (Byte Pair Encoding)
  Pretrained available here : https://bpemb.h-its.org</br>
  This paper using Indonesian Embedding (100 Dimension)</br>
- GloVe
  Pretrained available here : https://nlp.stanford.edu/projects/glove/ </br>
  This paper using Twitter Word Vector (200 Dimension). For Information, GloVe was training on large dataset containing multi-language</br>
### BERT
- indolem/indobertweet-base-uncased</br>
- indolem/indobert-base-uncased</br>

All BERT pretained model available on huggingface. You can try with different pretrained model
### Setup
All Classification Stages above was trained on M1 GPU using Tensorflow and PyTorch (BERT). 3 models that is Fasttext, BPE and GloVe using bidirectional LSTM architecture.</br>

# Topic Detection Techniques

### Algorithm
- LDA
- LSA
- HDP

# Classification Results
## Stage 1 (2 Classes | Others vs non-Others)
| Model                                | Macro Avg. Precision | Macro Avg. Recall | Macro Avg. F1 | Accuracy |
|--------------------------------------|----------------------|-------------------|---------------|----------|
| BoW                                  | 0.62                 | 0.71              | 0.62          | 0.72     |
| TF-IDF                               | 0.68                 | 0.71              | 0.69          | 0.77     |
| 2-Grams                              | 0.61                 | 0.63              | 0.62          | 0.72     |
| 3-Grams                              | 0.62                 | 0.62              | 0.62          | 0.70     |
| Fasttext (Pretrained: Indonesian)    | 0.67                 | 0.66              | 0.66          | 0.72     |
| BPE (Pretrained: Indonesian)         | 0.68                 | 0.69              | 0.69          | 0.75     |
| GloVe (Pretrained: Twitter Multilingual) | 0.63              | 0.62              | 0.63          | 0.71     |
| BERT: indobert-base-uncased          | 0.63                 | 0.58              | 0.58          | 0.70     |
| BERT: indobertweet-base-uncased      | 0.78                 | 0.71              | 0.73          | 0.81     |


## Stage 2 & 3 (7 Classes)

| Model                                | Macro Avg. Precision |       | Macro Avg. Recall |       | Macro Avg. F1 |       | Accuracy |       |
|--------------------------------------|----------------------|-------|-------------------|-------|---------------|-------|----------|-------|
|                                      | Original Data        | Original + Augmented Data | Original Data | Original + Augmented Data | Original Data | Original + Augmented Data | Original Data | Original + Augmented Data |
| BoW                                  | 0.45                 | 0.85  | 0.49              | 0.85  | 0.46          | 0.85  | 0.61     | 0.85  |
| TF-IDF                               | 0.61                 | 0.85  | 0.69              | 0.86  | 0.64          | 0.85  | 0.71     | 0.86  |
| 2-Grams                              | 0.46                 | 0.79  | 0.70              | 0.82  | 0.49          | 0.80  | 0.63     | 0.80  |
| 3-Grams                              | 0.30                 | 0.67  | 0.64              | 0.76  | 0.31          | 0.68  | 0.48     | 0.68  |
| Fasttext (Pretrained: Indonesian)    | 0.60                 | 0.80  | 0.63              | 0.81  | 0.60          | 0.80  | 0.72     | 0.81  |
| BPE (Pretrained: Indonesian)         | 0.58                 | 0.83  | 0.58              | 0.82  | 0.57          | 0.82  | 0.66     | 0.83  |
| GloVe (Pretrained: Twitter-Multilingual) | 0.63              | 0.85  | 0.61              | 0.59  | 0.85          | 0.58  | 0.85     | 0.85  |
| BERT: indobert-base-uncased          | 0.56                 | 0.88  | 0.61              | 0.87  | 0.57          | 0.87  | 0.67     | 0.87  |
| BERT: indobertweet-base-uncased      | 0.75                 | 0.91  | 0.76              | 0.91  | 0.73          | 0.91  | 0.80     | 0.92  |

# Topic Detection Results
...

# Paper & Cite
Our research can be accessed through this website below:
* ...
```bibtex
@misc{..,
  doi = {...},
  url = {...},
  author = {Indrabayu, Ardia Effendy and Basuki},
  keywords = {BERT, GPT-4, Mental Health, Topic Detection, Word-Embedding},
  title = {Automatic Categorization of Mental Health Frame in Indonesian X (Twitter) Text using Classification and Topic Detection Techniques},
  publisher = {Khazanah},
  year = {2024},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

  
