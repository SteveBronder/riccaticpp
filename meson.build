project('riccaticpp', 'cpp', version: '1.0.0')

riccaticpp_dep = declare_dependency(
  include_directories: include_directories('include'),
)

# pybind11 binding source, exposed for parent projects that build their own
# Python extension against riccaticpp (e.g. scipy's pyriccaticpp, which
# wires its own np_dep / pybind11_dep / eigen3_dep alongside this).
pymain_sources = files('src/pymain.cpp')
