# Factorization Machine (FM)

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 1.x](https://img.shields.io/badge/TensorFlow-1.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A TensorFlow implementation of Factorization Machines for classification tasks, based on the original paper by Steffen Rendle (2010).

<img width="864" height="432" alt="image" src="https://github.com/user-attachments/assets/8aeb8ca8-ad21-46fa-a95c-8e12cfa70166" />

## Abstract

Factorization Machines (FMs) are a general predictor that combines the advantages of Support Vector Machines (SVMs) with factorization models. Unlike SVMs, FMs model all nested interactions between variables using factorized parameters, which enables them to estimate reliable parameters even under high sparsity. This implementation provides an efficient FM classifier suitable for both dense and sparse high-dimensional data.

## Key Features

- **Efficient Pairwise Interactions**: Models all pairwise feature interactions in O(kn) time complexity
- **Sparsity Handling**: Excellent performance on sparse, high-dimensional datasets
- **Flexible Architecture**: Configurable embedding dimensions and regularization
- **Batch Normalization**: Stable training with input normalization
- **Dropout Regularization**: Prevents overfitting
- **Multi-class Support**: Native support for multi-class classification

## Model Architecture

The Factorization Machine models the target variable as:

<img width="299" height="45" alt="image" src="https://github.com/user-attachments/assets/95f3fe8a-22f2-4108-8145-bb4db9bfe4e0" />


Where:
- `w₀` is the global bias
- `wᵢ` are the linear weights
- `vᵢ` are the latent factor vectors
- `⟨vᵢ,vⱼ⟩` models the interaction between features i and j

The bi-interaction pooling layer efficiently computes pairwise interactions using:

<img width="214" height="47" alt="image" src="https://github.com/user-attachments/assets/54020796-7246-4b4c-b177-7bd3b80cc23a" />


## Parameters

1. **inp_dim (int):** Input feature dimension
2. **emb_dim (int, default=8):** Embedding dimension for latent factors
3. **n_classes (int, default=2):** Number of output classes
4. **keep_prob (float, default=0.8):** Dropout keep probability
5. **use_gpu (bool, default=False):** Whether to use GPU acceleration

## Use Cases

- Recommendation Systems: User-item interactions with side features
- Click-Through Rate Prediction: Sparse categorical features
- Natural Language Processing: Text classification with high-dimensional features
- Biological Data Analysis: Gene expression and protein interaction prediction

## Theoretical Background

### Advantages over Linear Models

- Captures feature interactions without manual feature engineering
- Handles high-dimensional sparse data effectively
- Non-linear decision boundaries through pairwise interactions

### Advantages over Kernel Methods

- Linear time complexity for prediction
- No need for sophisticated kernel functions
- Direct learning of interaction strengths

## References

> 1. Rendle, S. (2010). Factorization Machines. In 2010 IEEE International Conference on Data Mining.
> 2. Rendle, S. (2012). Factorization Machines with libFM. ACM Transactions on Intelligent Systems and Technology.

---

<br>

<h2 align="center">✨ Author</h2>

<p align="center">
  <b>Saad Abdur Razzaq</b><br>
 Machine Learning Engineer | Effixly AI
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/saadarazzaq" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn">
  </a>
  <a href="mailto:sabdurrazzaq124@gmail.com">
    <img src="https://img.shields.io/badge/Email-grey?style=for-the-badge&logo=gmail" alt="Email">
  </a>
  <a href="https://saadarazzaq.dev" target="_blank">
    <img src="https://img.shields.io/badge/Website-black?style=for-the-badge&logo=google-chrome" alt="Website">
  </a>
</p>

<br>
