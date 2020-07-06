# Imperial College London - Dissertation
Spatial Correlation Models of Earthquake Intensity Measures

Background:
A key part of the evaluation of the potential earthquake risk to the building stock is the estimation of the Hazard in a study region. Calculations of Earthquake Hazards is done within a Probabilistic framework called Probabilistic Seismic Hazard Analysis, or PSHA. PSHA essentially draws from datasets of previous earthquake events and geologic information about existing regional faults to develop Probability Density Functions that are then broadly used in a Monte Carlo simulation to calculate the probability of exceedance of a specific intensity measure at a given site or given portfolio of spacially distanced sites. Probability density functions are build to estimate the seismicity of seperate raptures near the study region and for the intensity measure estimations given specific seismic events occur. The latter one is done using a Ground Motion Model, an example of which is Chiou and Youngs 2014 paper implemented in the main Python code.

File No1: ChiouAndYoungs2014

Ground Motion Models:
Ground motion model give the lognormal distribution of Intensity measures given a specific set of parameters. They have a significant aleatory variability that is expressed via within and between event residuals that are standard normal variates N(0,1). 

Intesity Measure Fields for Portfolios:
When an estimate of the intesity measure fields are needed for a portfolio of buildings that are located at different positions then a random field needs to be simulated using the mean and realziations of the residuals at descrete points.

Correlations:
Due to strong correlations between residuals at points that are nearby a correlation function of the field needs to be developed. In the specific project this is done using the Jayaram & Baker 2009 correlation model. Once the Mean and Standard Deviations and the Correlations are developed the random field could be constructed.

File No2: MRB

Problem Discretization:
Due to computation time and memory constraints for large projects an approximation of the random field needs to be made. The study region is discretized into subregions and the intensity measures are sampled at the mean points of the discretized regions. However, the standard deviations and the correlation matrices are altered to reflect the error introduced by the discretization. This is the main problem in the Dissertation.

Calculations of the effective STD and Correlation:
This is done using the estimation of a quadruple integral. Which is done by Monte Carlo simulations. My intial model suggest that 100 data points would be roughly enough to get an estimate of the correlation.



