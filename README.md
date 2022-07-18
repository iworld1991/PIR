<!-- #region -->
# Perceived Income Risks 
- Author: Tao Wang
- Stage: work in progress. Preliminary. 

## To-knows 

- To download the entire working repo, go to the shell, navigate to the desirable locaiton, and type 
  - `git init`
  - `git clone https://github.com/iworld1991/PIR.git`.   
- For the most recent draft, see [here](./PIR.pdf).
- For replication of all the work, navigate [here](/WorkingFolder/PythonCode/) and run "do_all.py'' by doing the following in the shell: 
  - `ipython do_all.py`
- For reproducing latex and pdf draft after replication, navigate to the main working directoary, and run the "to_tex.sh" by typing the following in the shell: 
  - `source ./to_tex.sh`
  
  
## Structure of codes

## Empirics

1. [Density estimation of the survey answers](./WorkingFolder/PythonCode/DoDensityEst.ipynb) that draws from the general code [DensityEst](./WorkingFolder/PythonCode/DensityEst.py)
   
2. [Income risks decomposition](./WorkingFolder/PythonCode/IncomeRisksEst.ipynb) that draws on the [income process class](./WorkingFolder/PythonCode/IncomeProcess.ipynb)
   - [Estimation allowing for infrequent shocks](IncomeRisksInfrequentEst.ipynb) 

3. [Micro empirical analysis](./WorkingFolder/PythonCode/MicroRiskProfile.ipynb) on the cross-sectional heterogeneity of perceived risks (PR)

4. [Macro empirical analysis](./WorkingFolder/PythonCode/MacroRiskProfile.ipynb) on how PR correlate with macroeconomic conditions, i.e. labor market tightness

## Model


1.  [SCFMoments](./WorkingFolder/PythonCode/SCFData.ipynb)
2. [Parameters](./WorkingFolder/PythonCode/PrepareParameters.ipynb)
  - stored as a dictionary, to be directly imported into model notebooks


1. [Life-cycle model with Markov state of belief](./WorkingFolder/PythonCode/SolveLifeCycleMABelief.ipynb) 
   -  [Baseline model without belief state](./WorkingFolder/PythonCode/SolveLifeCycleMAShock.ipynb)
   - [cross-validating the solutions with HARK](./WorkingFolder/PythonCode/SolveLifeCycleMABelief-ComparisonHARK.ipynb)
   - [code-testing](./WorkingFolder/PythonCode/SolveLifeCycleMABelief-Test.ipynb)
   - a model [extension](./WorkingFolder/PythonCode/SolveLifeCycle-DC.ipynb) to allow adjustment cost of consumption (under development)

2. [Aggregate dynamics and GE of the life cycle economy](./WorkingFolder/PythonCode/OLG-GE-MAShock.ipynb) (no aggregate risks)

   - a possible extension to allow aggregate risks (Krusell-Smith & life-cycle economy)

3. [Markov regime switching model of subjective PR](./WorkingFolder/PythonCode/SubjectiveProfileEst.ipynb)


## Supporting utilities/resources 



1. [Utility](Utility.ipynb)
   
2. [Latex table manipulation](./WorkingFolder/PythonCode/TexTablesMover.ipynb)
<!-- #endregion -->

```python

```
