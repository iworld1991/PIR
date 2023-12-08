<!-- #region -->
# Perceived versus Calibrated Income Risks in Heterogeneous-agent Consumption Models 
- Originally circulated with the title "Perceived Income Risks"
- Author: Tao Wang

## To-knows

- To download the entire working repo, go to the shell, navigate to the desirable locaiton, and type
  - `git init`
  - `git clone https://github.com/iworld1991/PIR.git`.   
- For the most recent draft, see [here](./PIR.pdf).


## Structure of the code for replicating the results of the paper

## Empirics

1. [Income risks decomposition](./WorkingFolder/PythonCode/IncomeRisksEst.ipynb) that draws on the [income process class](./WorkingFolder/PythonCode/IncomeProcess.ipynb)
 
2. [Micro empirical analysis](./WorkingFolder/PythonCode/MicroRiskProfile.ipynb) on the cross-sectional heterogeneity of perceived risks (PR)

## Model

### Calibration
1.  [Household Wealth Data from SCF](./WorkingFolder/PythonCode/SCFData.ipynb)
2. [Parameters](./WorkingFolder/PythonCode/PrepareParameters.ipynb): stored as a dictionary, to be directly imported into model notebooks
3. [Calibration of the heterogeneous wage/unemployment risks from SCE](./WorkingFolder/PythonCode/HeterogeneousRisksEstMain.ipynb)

### Model
1. [Life-cycle consumption/saving model with permanent/persistent/transitory income risks](./WorkingFolder/PythonCode/SolveLifeCycle.ipynb)
 

2. [Stationary distribution and GE of the life cycle economy producing the baseline result](./WorkingFolder/PythonCode/OLG-GE.ipynb) (no aggregate risks)
   - [an extended version with heterogeneous income risks and growth rates for model experiment results](./WorkingFolder/PythonCode/OLG-GE-HetroTypes.ipynb)

3. [Compare model results and data](./WorkingFolder/PythonCode/PlotModelResults.ipynb) 


## Other derivative results (not in the main body of the paper)

### Empirics 
1. [Density estimation of the survey answers](./WorkingFolder/PythonCode/DoDensityEst.ipynb) that draws from the general code [DensityEst](./WorkingFolder/PythonCode/DensityEst.py)
2. [Evidence for infrequent shocks to monthly wage growth](./WorkingFolder/PythonCode/IncomeRisksInfrequentEst.ipynb) 

#### Model 
1. Life cycle problem sovlver
   - [cross-validating the solutions with HARK](./WorkingFolder/PythonCode/SolveLifeCycleBelief-ComparisonHARK.ipynb)
   - [code-testing](./WorkingFolder/PythonCode/SolveLifeCycleBelief-Test.ipynb)
   - a model [extension](./WorkingFolder/PythonCode/SolveLifeCycle-DC.ipynb) to allow adjustment cost of consumption (under development)
2. [Markov regime switching model of subjective PR](./WorkingFolder/PythonCode/SubjectiveProfileEst.ipynb) which draws from [model class](./WorkingFolder/PythonCode/MarkovSwitchingEst.ipynb)
3. [An extended life-cycle model with a Markov belief state](./WorkingFolder/PythonCode/SolveLifeCycleBelief.ipynb)
3. [An extended OLG-GE model with state-dependent beliefs of income risks](./WorkingFolder/PythonCode/OLG-GE-Belief.ipynb)


## Supporting utilities/resources


1. [Utility](./WorkingFolder/PythonCode/Utility.ipynb)

2. [Latex table manipulation](./WorkingFolder/PythonCode/TexTablesMover.ipynb)
<!-- #endregion -->

```python

```
