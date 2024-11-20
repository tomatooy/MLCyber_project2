# KOV

The entry file is KOV.py

# Kov Algorithm: Summary and Implementation
### Objective
The NGCG algorithm optimizes adversarial suffixes by balancing two objectives:

- **Negative log-likelihood (NLL)** to induce harmful behavior.
- **Log-perplexity** to maintain natural language coherence.

### Loss Function
The loss function is defined as:

```math
L(x_{1:n}) = - \log p(x^*_{n+1:H} \mid x_{1:n}) 
+ \lambda \left( -\frac{1}{n-1} \sum_{i=2}^{n} \log p(x_i \mid x_{1:i-1}) \right)