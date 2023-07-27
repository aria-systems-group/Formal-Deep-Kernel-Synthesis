# Formal-Deep-Kernel-Synthesis

This code makes use of the α,β-CROWN verifier which is developed by a team from CMU, UCLA, Drexel University, Columbia University and UIUC [1, 2, 3, 4]. The source code can be found here: https://github.com/Verified-Intelligence/alpha-beta-CROWN, their README file has been included in the folder alpha-beta-CROWN/.

Some modifications have been made from their main branch to produce needed outputs, hence the inclusion of the folder here.


This also makes use of the PosteriorBounds.jl repo [5], which is automatically added to the Julia package manager via juliacall.

## Installation

This code is run in a miniconda environment that will install all the necessary packages during setup. If you don't have conda, you can install [miniconda](https://docs.conda.io/en/latest/miniconda.html).


```bash
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name deep-kernel-syn
# install all dependents into the deep-kernel-syn environment
conda env create -f alpha-beta-CROWN/complete_verifier/environment.yml --name deep-kernel-syn
# activate the environment
conda activate deep-kernel-syn
```

### Julia Tools
Several packages must be installed in Julia for this code, these can be installed by opening julia and the entering the package manager with ].
They should be added inside the conda environment.

```bash
pkg> add JuMP, Ipopt, PyCall, SpecialFunctions, Plots, IterTools, ProgressBars, JLD, Distributions
pkg> add https://github.com/aria-systems-group/PosteriorBounds.jl
```

### BMDP Tool
This package depends on the `bmdp-tool` here: https://github.com/aria-systems-group/bmdp-tool

The tool should be compiled using Make and the `synthesis` executable moved to a location on the user executable path e.g. `/usr/local/bin`.



## Citation 
[1] Xu, H. Zhang, S. Wang, Y. Wang, S. Jana, X. Lin, and C.-J. Hsieh,
“Fast and Complete: Enabling complete neural network verification
with rapid and massively parallel incomplete verifiers,” in International
Conference on Learning Representations, 2021. [Online]. Available:
https://openreview.net/forum?id=nVZtXBI6LNn

[2] S. Wang, H. Zhang, K. Xu, X. Lin, S. Jana, C.-J. Hsieh, and J. Z. Kolter,
“Beta-CROWN: Efficient bound propagation with per-neuron split
constraints for complete and incomplete neural network verification,”
Advances in Neural Information Processing Systems, vol. 34, 2021.

[3] H. Zhang, T.-W. Weng, P.-Y. Chen, C.-J. Hsieh, and L. Daniel,
“Efficient neural network robustness certification with general
activation functions,” Advances in Neural Information Processing
Systems, vol. 31, pp. 4939–4948, 2018. [Online]. Available: https://arxiv.org/pdf/1811.00866.pdf

[4] H. Zhang, S. Wang, K. Xu, L. Li, B. Li, S. Jana, C.-J. Hsieh, and
J. Z. Kolter, “General cutting planes for bound-propagation-based neural network verification,” Advances in Neural Information Processing
Systems, 2022.
