<!-- #region -->
# What Explains the Heterogeneity in Perceived Income Risks?
- Derivative project from "Perceived Income Risks"
- Author: Tao Wang


## To-do 
-  Commuting zone/local labor market outcomes
-  Aggregate realizations from SF Fed
-  Historic vols from SIPP instead of PSID

## To-knows

- To download the entire working repo, go to the shell, navigate to the desirable locaiton, and type
  - `git init`
  - `git clone https://github.com/iworld1991/PIR.git`.   


## Structure of the Python code

## Empirics

1. [Density estimation of the survey answers](./WorkingFolder/PythonCode/DoDensityEst.ipynb) that draws from the general code [DensityEst](./WorkingFolder/PythonCode/DensityEst.py)

3. [Micro empirical analysis](./WorkingFolder/PythonCode/MicroRiskProfile.ipynb) on the cross-sectional heterogeneity of perceived risks (PR)

4. [Macro empirical analysis](./WorkingFolder/PythonCode/MacroRiskProfile.ipynb) on how PR correlate with macroeconomic conditions, i.e. labor market tightness
