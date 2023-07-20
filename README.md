# Template Recovery Attack on Homomorphically Encrypted Biometric Recognition Systems with Unprotected Threshold Comparison

## Description

This repository contains the implementation of a non-adaptive biometric template recovery attack that exploits the cleartext score disclosure vulnerability of HE-based BTPs that compute an inner product-based similarity measure under encryption to compare their encrypted templates.
This attack requires no training and a few random fake templates with their corresponding scores, from which the unprotected target template is recovered using the Lagrange multiplier optimization method.
The evaluation of this attack is twofold:

- Verification of whether the recovered template is deemed similar to the target template held by recognition systems set to accept 0.1%, 0.01%, and 0.001% FMR.
- Estimation of the number of fake templates and their corresponding scores leading to a template recovery with a 100% success rate.

### What is the risk of revealing cleartext scores?

Once a target biometric template is recovered, it cannot be mitigated.
Even if this template is replaced by a freshly generated template, the recovered template and its neighboring embeddings can still fool the system since they are close in the embedding space.
To prevent this, the leakage should be limited by carrying out the comparison with the threshold into the encrypted domain so that only the decision is delivered encrypted.

## Dependencies

This is a Python 3.9 implementation that requires the following packages:

- [`NumPy`](https://numpy.org/)  
- [`SciPy`](https://scipy.org/)

## Datasets

From the [VGGFace2](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8373813) dataset or the [LFW](https://inria.hal.science/inria-00321923/document) dataset, we extract target facial feature vectors of dimension 512 using [ResNet-100](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) trained with the  [ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf) loss.

## Experiments

The following experiments evaluate the template recovery attack for different quantization approaches.

1) Run `matedComparisonOrigRec.py` to evaluate the mated score distribution of original pairs `IP(original, original)` vs. cross pairs `IP(original, recovered)` and measures the attack success rate for different quantization approaches.
2) Run `upperBoundAttackIP.py` to estimate the number of fake templates for the case of revealed IP scores without quantization.
3) Run `upperBoundAttackPre.py` to estimate the number of fake templates for the case of revealed IP scores with Precision-based quantization [[B18]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8698601).
4) Run `upperBoundAttackTab.py` to estimate the number of fake templates for the case of revealed IP scores with Table-based quantization [[BHVP22]](https://ieeexplore.ieee.org/abstract/document/10007958)
.

## References

[[B18]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8698601)
[[BHVP22]](https://ieeexplore.ieee.org/abstract/document/10007958)

## Bibtex Citation

```
This is an accepted paper at the IJCB 2023 Conference. The proper citation will be made available soon.
```
