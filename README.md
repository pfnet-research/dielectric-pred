# Dielectric Tensor Prediction for Inorganic Materials Using Latent Information from Preferred Potential


## Related information
**Implementation of "Dielectric Tensor Prediction for Inorganic Materials Using Latent Information from Preferred Potential"**

Zetian Mao, Wenwen Li, Jethro Tan. The paper is available at [https://arxiv.org/abs/2405.09052](https://arxiv.org/abs/2405.09052).

The base architecture Preferred Potential (PFP) is a commercial model, see details at [https://matlantis.com/news/pfp-validation-for-public-v5-0-0](https://matlantis.com/news/pfp-validation-for-public-v5-0-0).

The paper title "[Towards universal neural network potential for material discovery applicable to arbitrary combination of 45 elements](https://doi-org.utokyo.idm.oclc.org/10.1038/s41467-022-30687-9)" reports related information about PFP.

## Introduction

![overview](/imgs/overview.png)

**Abstract**: Dielectrics are materials with widespread applications in flash memory, central processing units, photovoltaics, capacitors, etc. However, the availability of public dielectric data remains limited, hindering research and development efforts. Previously, machine learning models focused on predicting dielectric constants as scalars, overlooking the importance of dielectric tensors in understanding material properties under directional electric fields for material design and simulation. This study demonstrates the value of common equivariant structural embedding features derived from a universal neural network potential in enhancing the prediction of dielectric properties. To integrate channel information from various-rank latent features while preserving the desired SE(3) equivariance to the second-rank dielectric tensors, we design an equivariant readout decoder to predict the total, electronic, and ionic dielectric tensors individually, and compare our model with the state-of-the-art models. Finally, we evaluate our model by conducting virtual screening on thermodynamical stable structure candidates in Materials Project. The material Ba<sub>2</sub>SmTaO<sub>6</sub> with large band gaps (E<sub>g</sub>=3.36eV) and dielectric constants (ϵ=93.81) is successfully identified out of the 14k candidate set. The results show that our methods give good accuracy on predicting dielectric tensors of inorganic materials, emphasizing their potential in contributing to the discovery of novel dielectrics.

## Example

An example of our implementation is available at [notebooks/dielectric_tensor.ipynb](notebooks/dielectric_tensor.ipynb).
