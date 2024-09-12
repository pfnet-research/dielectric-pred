# Dielectric Tensor Prediction for Inorganic Materials Using Latent Information from Preferred Potential


## Related information
**Implementation of "Dielectric Tensor Prediction for Inorganic Materials Using Latent Information from Preferred Potential"**

Zetian Mao, Wenwen Li, Jethro Tan. The paper is available at [https://arxiv.org/abs/2405.09052](https://arxiv.org/abs/2405.09052).

The base architecture Preferred Potential (PFP) is a commercial model, see details at [https://matlantis.com/news/pfp-validation-for-public-v5-0-0](https://matlantis.com/news/pfp-validation-for-public-v5-0-0).

The paper title "[Towards universal neural network potential for material discovery applicable to arbitrary combination of 45 elements](https://doi-org.utokyo.idm.oclc.org/10.1038/s41467-022-30687-9)" reports related information about PFP.

## Introduction

![overview](/imgs/overview.png)

**Abstract**: 
Dielectrics are crucial for technologies like flash memory, CPUs, photovoltaics, and capacitors, but public data on these materials are scarce, restricting research and development. Existing machine learning models have focused on predicting scalar polycrystalline dielectric constants, neglecting the directional nature of dielectric tensors essential for material design. This study leverages multi-rank equivariant structural embeddings from a universal neural network potential to enhance predictions of dielectric tensors. We develop an equivariant readout decoder to predict total, electronic, and ionic dielectric tensors while preserving O(3) equivariance, and benchmark its performance against state-of-the-art algorithms. Virtual screening of thermodynamically stable materials from Materials Project for two discovery tasks, high-dielectric and highly anisotropic materials, identifies promising candidates including Cs<sub>2</sub>Ti(WO<sub>4</sub>)<sub>3</sub> (band gap E<sub>g</sub> = 2.93eV, dielectric constant ε = 180.90) and CsZrCuSe<sub>3</sub> (anisotropic ratio α<sub>r</sub> = 121.89). The results demonstrate our model’s accuracy in predicting dielectric tensors and its potential for discovering novel dielectric materials.

## Example

An example of our implementation is available at [notebooks/dielectric_tensor.ipynb](notebooks/dielectric_tensor.ipynb).

We evaluated our method on the Matbench dataset, see [notebooks/matbench.ipynb](notebooks/matbench.ipynb) for details.
