# MSc-pre-process
Pre processing of Te, ne, Ti, Vt data for input on MARS-F

This is my first repository and it comes a while after the beginning of the project.

This repository includes the code used to pre-process the data for input into MARS-F. Will be pre-processing the following:
- Te - Electron Temperature data from Thomson Scattering experiments
- ne - Electron density data from TS experiments
- Ti - Ion tmeperature data from charge exchange spectroscopy
- Vt - rotation velocity from ...

Want to express all these quantities as a somewhat smooth function of normalised flux coordinate Psi_N and time t

Will be doing stuff such as:
- importing data from freia
- removing extremums from data
- cleaning out infinities and NaNs
- lots of interpolation to have data objects match nicely to each other
- some reconstructing of data if data is incomplete
- n'th order bivariate polynomial fitting (hopefully piecewise too)
- lots o' data visualisation
