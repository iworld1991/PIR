<!-- #region -->
# Perceived Income Risks 
- Author: Tao Wang
- Stage: work in progress. Preliminary. 

## To-knows 

- To download the entire working repo, go to the shell, navigate to the desirable locaiton, and type 
  - `git init`
  - `git clone https://github.com/iworld1991/IncExpProject.git`.   
- For the most recent draft, see [here](./PIR.pdf).
- For replication of all the work, navigate [here](/WorkingFolder/PythonCode/) and run "do_all.py'' by doing the following in the shell: 
  - `ipython do_all.py`
- For reproducing latex and pdf draft after replication, navigate to the main working directoary, and run the "to_tex.sh" by typing the following in the shell: 
  - `source ./to_tex.sh`
  
  
## Structure of codes

## Empirics

1. [Density estimation of the survey answers](./WorkingFolder/PythonCode/DensityEst.ipynb) (under test)
   
2. [Income risks decomposition](./WorkingFolder/PythonCode/IncomeRisksEst.ipynb) that draws on the [income process class](./WorkingFolder/PythonCode/IncomeProcess.ipynb)

3. [Micro empirical analysis](./WorkingFolder/PythonCode/MicroRiskProfile.ipynb) on the cross-sectional heterogeneity of perceived risks (PR)

4. [Macro empirical analysis](./WorkingFolder/PythonCode/MacroRiskProfile.ipynb) on how PR correlate with macroeconomic conditions, i.e. labor market tightness

## Model

1. [Parameters](./WorkingFolder/PythonCode/PrepareParameters.ipynb)

   - prepare the baseline parameters, stored in a txt file. 


1. [Life-cycle model](./WorkingFolder/PythonCode/SolveLifeCycle.ipynb) 
   - [cross-validating the solutions with HARK](./WorkingFolder/PythonCode/SolveLifeCycle-ComparisonHARK.ipynb)
   - a model [extension](./WorkingFolder/PythonCode/SolveLifeCycle-DC.ipynb) to allow adjustment cost of consumption (under development)

2. [Aggregate dynamics and GE of the life cycle economy](./WorkingFolder/PythonCode/LifeCycle-AggregateDynamics.ipynb) (no aggregate risks)

   - a possible extension to allow aggregate risks (Krusell-Smith & life-cycle economy)

3. [Markov regime switching model of subjective PR](./WorkingFolder/PythonCode/SubjectiveProfileEst.ipynb)


## Supporting utilities/resources 



1. [Utility](Utility.ipynb)
   - used for solving the [life-cycle model](./WorkingFolder/PythonCode/SolveLifeCycle.ipynb)
   
2. [Latex table manipulation](./WorkingFolder/PythonCode/TexTablesMover.ipynb)
<!-- #endregion -->
