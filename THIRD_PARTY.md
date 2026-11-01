# Third-party attribution

`src/virtual_staining/third_party/allen_fnet.py` contains the parameterised 3D
network from [AllenCellModeling/pytorch_fnet](https://github.com/AllenCellModeling/pytorch_fnet),
commit `fd8fc92cc4c43d9d82da1c66c7e17e2447dd0830`, original path
`fnet/nn_modules/fnet_nn_3d_params.py`. Its algorithm and state-dict names are
unchanged. A provenance header was added. Copyright 2020 Gregory R Johnson,
Allen Institute. The complete upstream Allen Institute Software License is
included alongside the module. It permits non-commercial use subject to its
terms; commercial use requires the Institute's permission. This notice applies
to the attributed component, not an invented licence for the entire project.

The accompanying new data adapter and trainer are independently implemented.
The reference architecture is attributed, not claimed as our invention.

Method: Ounkomol et al., Nature Methods (2018),
https://doi.org/10.1038/s41592-018-0111-2. The paper's `release_1` is pinned at
`e1f51f50a641af048f76d71c9ce03a740ab33195`. With one input and one output,
the later parameterised module implements the same recursive structure. A
compatibility check against the pinned original is required before using the
term reproduction. A reduced-width pilot is explicitly a different model.
