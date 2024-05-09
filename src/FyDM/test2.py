"""Created on May 09 10:19:40 2024"""
from src.FyDM.__backend.fdm_ import OneDimensionalFDM

p = OneDimensionalFDM.from_yaml('./test.yaml')
print(p.pde_properties)
