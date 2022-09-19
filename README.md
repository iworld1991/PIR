<!-- #region -->
# Perceived Income Risks
- Author: Tao Wang
- Stage: work in progress. Preliminary.

## To-knows

- To download the entire working repo, go to the shell, navigate to the desirable locaiton, and type
  - `git init`
  - `git clone https://github.com/iworld1991/PIR.git`.   
- For the most recent draft, see [here](./PIR.pdf).


## Structure of code

## Empirics

1. [Density estimation of the survey answers](./WorkingFolder/PythonCode/DoDensityEst.ipynb) that draws from the general code [DensityEst](./WorkingFolder/PythonCode/DensityEst.py)

2. [Income risks decomposition](./WorkingFolder/PythonCode/IncomeRisksEst.ipynb) that draws on the [income process class](./WorkingFolder/PythonCode/IncomeProcess.ipynb)
   - [Estimation allowing for infrequent shocks](./WorkingFolder/PythonCode/IncomeRisksInfrequentEst.ipynb) 

3. [Micro empirical analysis](./WorkingFolder/PythonCode/MicroRiskProfile.ipynb) on the cross-sectional heterogeneity of perceived risks (PR)

4. [Macro empirical analysis](./WorkingFolder/PythonCode/MacroRiskProfile.ipynb) on how PR correlate with macroeconomic conditions, i.e. labor market tightness

## Model


1.  [SCFMoments](./WorkingFolder/PythonCode/SCFData.ipynb)
2. [Parameters](./WorkingFolder/PythonCode/PrepareParameters.ipynb)
  - stored as a dictionary, to be directly imported into model notebooks


1. [Life-cycle consumption/saving model with permanent/persistent/transitory income risks](./WorkingFolder/PythonCode/SolveLifeCycle.ipynb)
   - [cross-validating the solutions with HARK](./WorkingFolder/PythonCode/SolveLifeCycleBelief-ComparisonHARK.ipynb)
   - [code-testing](./WorkingFolder/PythonCode/SolveLifeCycleBelief-Test.ipynb)
   - a model [extension](./WorkingFolder/PythonCode/SolveLifeCycle-DC.ipynb) to allow adjustment cost of consumption (under development)

2. [Stationary distribution and GE of the life cycle economy](./WorkingFolder/PythonCode/OLG-GE.ipynb) (no aggregate risks)
   - [an extended version with heterogeneous income risks, growth rates and time preference](./WorkingFolder/PythonCode/OLG-GE-HeterTypes.ipynb)

3. [Calibration of the heterogeneous wage/unemployment risks from SCE](./WorkingFolder/PythonCode/HeterogeneousRisksEstMain.ipynb)

## Other derivative results (not in the main body of the paper)

1. [Markov regime switching model of subjective PR](./WorkingFolder/PythonCode/SubjectiveProfileEst.ipynb)
2. [An extended life-cycle model with a Markov belief state](./WorkingFolder/PythonCode/SolveLifeCycleBelief.ipynb)
3. [An extended OLG-GE model with state-dependent beliefs of income risks](./WorkingFolder/PythonCode/OLG-GE-Belief.ipynb)

## Supporting utilities/resources



1. [Utility](./WorkingFolder/PythonCode/Utility.ipynb)

2. [Latex table manipulation](./WorkingFolder/PythonCode/TexTablesMover.ipynb)
<!-- #endregion -->

```python

```
