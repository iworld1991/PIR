<!-- #region -->
# Data and Code for "Perceived versus Calibrated Income Risks in Heterogeneous-agent Consumption Models" 
- Originally circulated with the title "Perceived Income Risks"
- Author: Tao Wang

## To-knows

- To download the entire working repo, go to the shell, navigate to the desirable locaiton, and type
  - `git init`
  - `git clone https://github.com/iworld1991/PIR.git`.   
- For the most recent draft, see [here](./PIR.pdf).

## Data information 

This paper uses datasets drawn from the following public sources. 

 - United States Census Bureau. Survey of Income and Program Participation [SIPP](https://www.census.gov/programs-surveys/sipp.html): 2014 Panel Wave 1-4 (2014-2017 data), and 2018, 2019, 2020, 2021 waves
 - Federal Reserve Bank of New York. Survey of Consumer Expectations [SCE](https://www.newyorkfed.org/microeconomics/sce/background.html): 2013-2021 microdata 
 - Board of Governors of the Federal Reserve Board. Survey of Consumer Finances [SCF](https://www.federalreserve.gov/econres/scfindex.htm):  2016 vintage
 - Other Data
   - United States Census Bureau. Basic Monthly Current Population Survey [CPS](https://www.census.gov/data/datasets/time-series/demo/cps/cps-basic.html): job separation and finding rates calculated from CPS micro data for 2013-2022
   - Federal Reserve Bank of St Louis [Fred](https://fred.stlouisfed.org/): headline CPI inflation for 2013-2022
   - Social Services Administration. Actuarial Life Table [Life Table](https://www.ssa.gov/oact/STATS/table4c6.html): 2020

## Structure of the replication package 
- DoFile

	 This file contains all Stata do files that cleans the CPS data, SIPP data for income risk estimation, SCE micro data. 
- Graphs
   - Graphs\ind
   - Graphs\model
   - Graphs\pop
   - Graphs\sce
   - Graphs\sipp
   - Graphs\model\Belief
   - Graphs\pop\hist
- SurveyData 
	- SurveyData\SCE
- OtherData (See data sources above}
	- OtherData\sipp
- PythonCode 

  See section “Structure of the code and replication instructions” below for full detail.
  
  - PythonCode\data
  - PythonCode\parameters

- Tables
   Tables\ind
   Tables\latex
   Tables\model
   Tables\sipp


## Structure of the code and replication instructions

## Empirics

0. [Download SCE Individual Data](./WorkingFolder/PythonCode/DownloadSCE.ipynb)
1. [Density estimation of the survey answers](./WorkingFolder/PythonCode/DoDensityEst.ipynb) that draws from the general code [DensityEst](./WorkingFolder/PythonCode/DensityEst.py)

2. [Income risks decomposition](./WorkingFolder/PythonCode/IncomeRisksEst.ipynb) that draws on the [income process class](./WorkingFolder/PythonCode/IncomeProcess.ipynb)
   - [Evidence for infrequent shocks to monthly wage growth](./WorkingFolder/PythonCode/IncomeRisksInfrequentEst.ipynb) 

3. [Micro empirical analysis](./WorkingFolder/PythonCode/MicroRiskProfile.ipynb) on the cross-sectional heterogeneity of perceived risks (PR)

## Model

### Calibration
1.  [Household wealth stats from SCF used for model comparison](./WorkingFolder/PythonCode/SCFData.ipynb)
2. [Parameters](./WorkingFolder/PythonCode/PrepareParameters.ipynb): stored as a dictionary, to be directly imported into model notebooks
3. [Calibration of the heterogeneous wage/unemployment risks from SCE](./WorkingFolder/PythonCode/HeterogeneousRisksEstMain.ipynb)

### Model
1. [Life-cycle consumption/saving model with permanent/persistent/transitory income risks](./WorkingFolder/PythonCode/SolveLifeCycle.ipynb)
   - [cross-validating the solutions with HARK](./WorkingFolder/PythonCode/SolveLifeCycleBelief-ComparisonHARK.ipynb)
   - [code-testing](./WorkingFolder/PythonCode/SolveLifeCycleBelief-Test.ipynb)
   - [indirect estimation of preference parameters to match data moments](./WorkingFolder/PythonCode/OLG-PE-Estimation.ipynb)
   - [comparative statistics analysis](./WorkingFolder/PythonCode/OLG-PE-ComparativeStats.ipynb)

2. [Stationary distribution and GE of the life cycle economy](./WorkingFolder/PythonCode/OLG-GE.ipynb) (no aggregate risks)
   - [an extended version with heterogeneous income risks, growth rates and time preference](./WorkingFolder/PythonCode/OLG-GE-HetroTypes.ipynb)

3. [Compare model results and data](./WorkingFolder/PythonCode/PlotModelResults.ipynb) 


## Other derivative results (not in the main body of the paper)

1. [Markov regime switching model of subjective PR](./WorkingFolder/PythonCode/SubjectiveProfileEst.ipynb) which draws from [model class](./WorkingFolder/PythonCode/MarkovSwitchingEst.ipynb)
2. [An extended life-cycle model with a Markov belief state](./WorkingFolder/PythonCode/SolveLifeCycleBelief.ipynb)
3. [An extended OLG-GE model with state-dependent beliefs of income risks](./WorkingFolder/PythonCode/OLG-GE-Belief.ipynb)

## Supporting utilities/resources


1. [Utility](./WorkingFolder/PythonCode/Utility.ipynb)

<!-- #endregion -->

```python

```
