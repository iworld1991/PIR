# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # The research question
#
#
# "The devil is in higher moments." Even if two people share identical expected income and homogeneous preferences, different degrees of income risks still lead to starkly different decisions such as saving/consumption and portfolio choices. This is well understood in models in which agents are inter-temporally risk-averse, or prudent, and the risks associated with future marginal utility motivate precautionary motives. The same logic carries through to models in which capital income and portfolio returns are stochastic, and the risks of returns naturally become the center of asset pricing. Such behavioral regularities equipped with market incompleteness due to reasons such as imperfect insurance and credit constraints have also been the cornerstone assumptions used in the literature on heterogeneous-agent macroeconomics. 
#
# Economists have long utilized cross-sectional distributions of realized microdata to estimate the stochastic environments relevant to the agents' decision, such as the income process. And then in modeling the estimated risk profile is taken as parametric inputs and the individual shocks are simply drawn from the shared distributions. (See <cite data-cite="blundell_consumption_2008">(Blundell, et al. 2013)</cite> as an example.) But one assumption implicitly made when doing this is that the agents in the model perfectly understand thus agree on the income risk profile imposed on them. As shown by the actively developing literature on expectation formation, in particular, the mounting evidence on heterogeneity in economic expectations held by micro agents, this assumption seems to be too stringent. To the extent that agents make decisions based on their *respective* perceptions, understanding the *perceived* income risk profile and its correlation structure with other macro variables are the keys to explaining their behavior patterns.
#
# This paper's goal is to understand the question discussed above by directly shedding light on the subjective income profile using the recently available density forecasts of labor income surveyed by New York Fed's Survey of Consumer Expectation (SCE). What is special about this survey is that agents are asked to provide histogram-type forecasts of their earning growth over the next 12 months together with a set of expectational questions about the macroeconomy. It is at a monthly frequency and has a panel structure allowing for consecutive observations of the same household over a horizon of 12 months. When the individual density forecast is available, a parametric density estimation can be made to obtain the individual-specific subjective distribution. And higher moments reflecting the perceived income risks such as variance, as well as the asymmetry of the distribution such as skewness allow me to directly characterize the perceived risk profile without relying on external estimates from cross-sectional microdata. This provides the first-hand measured perceptions on income risks that are truly relevant to individual decisions.
#
# Empirically, I can immediately ask the following questions. 
#
# - How much heterogeneity is there across workers' perceived income risks? What factors, i.e. household income, demographics, and other expectations, are correlated with the subjective risks in both individual and macro level? 
#
# - To what extent to which this heterogeneity in perceptions align with the true income risks facing different population group, or at least partly attributed to perceptive differences due to heterogeneity in information and information processing, as discussed in many models of expectation formation?  
#    - If we treat the income risks identified from cross-sectional inequality by econometricians as a benchmark, to what extent are the risks perceived by the agents?
#       - If agents know more than econometricians about their individual earnings, should the perceived risks be lower than econometrician's estimates?
#       - Or actually, do agents, due to inattention or other information rigidity in learning about recently realized shocks, perceive the overall risk to be higher?
#
# - If the subjective income risk can be decomposed into components of varying persistence (i.e. permanent vs transitory) based on assumed income process, it is possible to characterize potential deviations of perceptive income process from some well defined rational benchmark.
#      - For instance, if agents overestimate their permanent income risks? 
#      - If agents overestimate the persistence of the income process? <cite data-cite="rozsypal_overpersistence_2017">(Rozsypal and Schlafmann, 2017)</cite>
#      - One step back, if the log-normality assumption of income progress consistent with the surveyed data. Or it has non-zero skewness? This can be jointly tested using higher moments of the density forecasts.  
#  
# - Finally, not just the process of earning itself, but also its covariance with macro-environment, risky asset returns, matter a great deal. For instance, if perceived income risks are counter-cyclical, it has important labor supply and portfolio implications. (<cite data-cite="guvenen2014nature">(Guvenen, et al. 2014)</cite>, <cite data-cite="catherine_countercyclical_2019">(Catherine, 2019)</cite>)
#
#  
# One of the key challenges when addressing these questions is to separately account for the differences in perceived risks driven by differences in underlying risk profiles, i.e. the "truth", and the rest driven by perceptive and informational heterogeneity. The most straightforward way seems to be to compare econometrician's external estimates of the income process using realized data and the perceived from the subjective survey. But this approach implicitly assumes that econometricians correctly specify the model of the income process and ignores the likely superior information problem discussed above. Therefore, in this paper, instead of simply assuming the external estimate by econometricians is the true underlying income process, I characterize the differences between perception and the true process by jointly recovering the process using realized data and expectations based on a particular well-defined theory of expectation formation. The advantage of doing this is that one does not need to make a stringent assumption about either agents' full rationality or econometricians' correctness of model specification. It allows econometricians to utilize the information from expectations to understand the true law of the system. This is in a similar spirit to <cite data-cite="guvenen_inferring_2014">(Guvenen, 2014)</cite>, although the author does not use expectation survey, the consumption choice as the additional input for the joint estimation. 
#  
#
# Theoretically, once I can document robustly some patterns of the perceived income risks profile, it can ben incorporated into an otherwise standard life-cycle model involving consumption/portfolio decisions to explore its macro implications. Ex-ante, one may conjecture a few of the following scenarios. 
#
#   - If the subjective risks or skewness is found to be negatively correlated with the risky market return or business cycles, this exposes agents to more risks than a non-state-dependent income profile. 
#
#   - If according to the subjective risk profile, the downside risks are highly persistent than typically assumed, then it is in line with the rare disaster idea.  
#
#   - The perceptual differences lead to differences in MPCs, which is a different mechanism from credit-constraints and noninsurance of risks. 
#
#      
# ##  Relevant literature and potential contribution 
#
# This paper is relevant to four lines of literature. First, the idea of this paper echoes with an old problem in the consumption insurance literature: 'insurance or information' (<cite data-cite="pistaferri_superior_2001">Pistaferri, 2001</cite>, <cite data-cite="kaufmann_disentangling_2009">Kaufmann and Pistaferri, 2009</cite>,<cite data-cite="meghir2011earnings">Meghir et al. 2011</cite>). In any empirical tests of consumption insurance or consumption response to income, there is always a worry that what is interpreted as the shock has actually already entered the agents' information set or exactly the opposite. For instance, the notion of excessive sensitivity, namely households consumption highly responsive to anticipated income shock, maybe simply because agents have not incorporated the recently realized shocks that econometricians assume so (<cite data-cite="flavin_excess_1988">Flavin,1988</cite>). Also, recently, in the New York Fed [blog](https://libertystreeteconomics.newyorkfed.org/2017/11/understanding-permanent-and-temporary-income-shocks.html), the authors followed a similar approach to decompose the permanent and transitory shocks. My paper shares a similar spirit with these studies in the sense that I try to tackle the identification problem in the same approach: directly using the expectation data and explicitly controlling what are truly conditional expectations of the agents making the decision. This helps economists avoid making assumptions on what is exactly in the agents' information set. What differentiates my work from other authors is that I focus on higher moments, i.e. income risks and skewness by utilizing the recently available density forecasts of labor income. Previous work only focuses on the sizes of the realized shocks and estimates the variance of the shocks using cross-sectional distribution, while my paper directly studies the individual specific variance of these shocks perceived by different individuals. This will become clear in Section \ref{perceived-income-process-in-progress}. 
#
# Second, this paper is inspired by an old but recently reviving interest in studying consumption/saving behaviors in models incorporating imperfect expectations and perceptions. For instance, <cite data-cite="rozsypal_overpersistence_2017">(Rozsypal and Schlafmann, 2017)</cite> found that households' expectation of income exhibits an over-persistent bias using both expected and realized household income from Michigan household survey. The paper also shows that incorporating such bias affects the aggregate consumption function by distorting the cross-sectional distributions of marginal propensity to consume(MPCs) across the population. <cite data-cite="carroll_sticky_2018">(Carroll et al. 2018)</cite> reconciles the low micro-MPC and high macro-MPCs by introducing to the model an information rigidity of households in learning about macro news while being updated about micro news. <cite data-cite="lian2019imperfect">(Lian, 2019)</cite> shows that an imperfect perception of wealth accounts for such phenomenon as excess sensitivity to current income and higher MPCs out of wealth than current income and so forth. My paper has a similar flavor to all of these works by exploring the behavioral implications of households' perceptive imperfection. The novelty of my paper lies in the primary on the implications of heterogeneity in perceived higher moments such as risks and skewness. Various theories of expectation formation have different predictions about the cross-sectional and dynamic patterns of perceived risks. I examine these predictions in this paper.   
#
# This paper also contributes to the literature studying expectation formation using subjective surveys. There has been a long list of "irrational expectation" theories developed in recent decades on how agents deviate from full-information rationality benchmark, such as sticky expectation, noisy signal extraction, least-square learning, etc. Also, empirical work has been devoted to testing these theories in a comparable manner (<cite data-cite="coibion2012can">(Coibion and Gorodnichenko, 2012)</cite>, <cite data-cite="fuhrer2018intrinsic">(Fuhrer, 2018)</cite>). But it is fair to say that thus far, relatively little work has been done on individual variables such as labor income, which may well be more relevant to individual economic decisions. Therefore, understanding expectation formation of the individual variables, in particular, concerning both mean and higher moments, will provide fruitful insights for macroeconomic modeling assumptions. 
#
# Lastly, the paper is indirectly related to the research that advocated for eliciting probabilistic questions measuring subjective uncertainty in economic surveys (<cite data-cite="manski_measuring_2004">(Manski, 2004)</cite>, <cite data-cite="delavande2011measuring">(Delavande et al. 2011)</cite>, <cite data-cite="manski_survey_2018">(Manski, 2018)</cite>). Although the initial suspicion concerning to people’s ability in understanding, using and answering probabilistic questions is understandable, <cite data-cite="bertrand_people_2001">(Bertrand and Mullainathan,2001)</cite> and other works have shown respondents have the consistent ability and willingness to assign a probability (or “percent chance”) to future events. <cite data-cite="armantier_overview_2017">(Armantier et al. 2017)</cite>  have a thorough discussion on designing, experimenting and implementing the consumer expectation surveys to ensure the quality of the responses. Broadly speaking, the advocators have argued that going beyond the revealed preference approach, availability to survey data provides economists with direct information on agents’ expectations and helps avoids imposing arbitrary assumptions. This insight holds for not only point forecast but also and even more importantly, for uncertainty, because for any economic decision made by a risk-averse agent, not only the expectation but also the perceived risks matter a great deal.
#

# + {"code_folding": [12], "hide_output": true}
## import libraries for inserting figures 

import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline 
from IPython.display import display, Image
import matplotlib.image as mpimg
import os
import pandas as pd

path = os.getcwd()


def in_ipynb():
    try:
        if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
            return True
        else:
            return False
    except NameError:
        return False

# Determine whether to make the figures inline (for spyder or jupyter)
# vs whatever is the automatic setting that will apply if run from the terminal
if in_ipynb():
    # %matplotlib inline generates a syntax error when run from the shell
    # so do this instead
    get_ipython().run_line_magic('matplotlib', 'inline') 
else:
    get_ipython().run_line_magic('matplotlib', 'auto') 
# -

# # Data, variables and density estimation
#
# ## Data
# The data used for this paper is from the core module of Survey of Consumer Expectation(SCE) conducted by the New York Fed, a monthly online survey for a rotating panel of around 1,300 household heads. The sample period in my paper spans from June 2013 to June 2018, in total of 60 months. This makes about 79000 household-year observations, among which around 53,000 observations provide non-empty answers to the density question on earning growth. 
#
# Particular relevant for my purpose, the questionnaire asks each respondent to fill perceived probabilities of their same-job-hour earning growth to pre-defined non-overlapping bins. The question is framed as "suppose that 12 months from now, you are working in the exact same [“main” if Q11>1] job at the same place you currently work and working the exact same number of hours. In your view, what would you say is the percent chance that 12 months from now: increased by x% or more?".
#
# As a special feature of the online questionnaire, the survey only moves on to the next question if the probabilities filled in all bins add up to one. This ensures the basic probabilistic consistency of the answers crucial for any further analysis. Besides, the earning growth expectation regarding exactly the same position, same hours and same location has two important implications for my analysis. First, the requirements help make sure the comparability of the answers across time and also excludes the potential changes in earnings driven by endogenous labor supply decisions, i.e. working for longer hours. Second, the earning expectations, risks and tail risks measured here are only conditional. It excludes either unemployment, i.e. likely a zero earning, or an upward movement in the job ladder, i.e. a different earning growth rate. Therefore, it does not fully reflect the labor income risk profile relevant to each individual. 
#
# In so far as we want to tease out the earning changes from voluntary decisions, moving to a different job position should actually not be included in earning expectations. Therefore only the exclusion of an unemployment scenario is relevant for my purpose in characterizing the labor income risks. But what is assuring me is that the bias due to omission of unemployment risk is unambiguous. We could interpret the moments of same-job-hour earning growth as an upper bound for the level of growth rate and a lower bound for the income risk. To put it in another way, the expected earning growth conditional on employment is higher than the unconditional one, and the conditional earning risk is lower than the unconditional one. At the same time, since SCE separately elicits the perceived probability of losing the current job for each respondent, I could adjust the measured labor income moments taking into account the unemployment risk. 
#
# ## Density estimation and variables 
#
# With the histogram answers for each individual in hand, I follow <cite data-cite="engelberg_comparing_2009">(Engelberg, Manskiw and Williams, 2009)</cite> to fit each of them with a parametric distribution accordingly for three following cases. In the first case when there are three or more intervals filled with positive probabilities, it will be fitted with a generalized beta distribution. In particular, if there is no open-ended bin on the left or right, then 2-parameter beta distribution is sufficient. If there is either open-ended bin with positive probability, since the lower bound or upper bound of the support needs to be determined, a 4-parameter beta distribution is estimated. In the second case, in which there are exactly 2 adjacent intervals with positive probabilities, it is fitted with an isosceles triangular distribution. In the third case, if there is only one positive-probability of interval only, i.e. equal to one, it is fitted with a uniform distribution. 
#
# I have a reason to discuss at length the exact procedures for density distribution. It is important for this paper's purpose because I need to make sure the estimation assumptions of the density distribution do not mechanically distort my cross-sectional patterns of the estimated moments. This is the most obviously seen in the tail risk measure, skewness. The assumption of log normality of income process, common in the literature (See again <cite data-cite="blundell_consumption_2008">(Blundell et al. 2008)</cite>), implicitly assume zero skewness, i.e. that the income increase and decrease are equally likely. This may not be the case in our surveyed density for many individuals. In order to account for this possibility, the assumed density distribution should be flexible enough to allow for different shapes of subjective distribution. Beta distribution fits this purpose well. Of course, in the case of uniform and isosceles triangular distribution, the skewness is zero by default. For those of you who may wonder, the fractions of the density answers fitted with the beta, uniform, and triangular distributions are, respectively, xxx, xxx, xxx in our sample.  
#
# Since the microdata provided in the SCE website already includes the estimated mean, variance and IQR by the staff economists following the exact same approach, I directly use their estimates for these moments. At the same time, for the measure of tail-risk, i.e. skewness, as not provided, I use my own estimates. I also confirm that my estimates and theirs for the first two moments are correlated with a coefficient of 0.9. 
#
# For all the moment's estimates, there are inevitably extreme values. This could be due to the idiosyncratic answers provided by the original respondent, or some non-convergence of the numeric estimation program. Therefore, for each moment of the analysis, I exclude top and bottom $5\%$ observations, leading to a sample size of around 45,000. 
#
# I also recognize what is really relevant to many economic decisions such as consumption is real income instead of nominal income. Thanks to the availability of inflation expectation and inflation uncertainty (also estimated from density question) can be used to convert nominal earning growth moments to real terms. In particular, the real earning growth rate is expected nominal growth minus inflation expectation. 
#
#
# \begin{eqnarray}
# \overline {\Delta y^{r}}_{i,t} = \overline\Delta y_{i,t} - \overline \pi_{i,t}
# \end{eqnarray}
#
# The variance associated with real earning growth, if we treat inflation and nominal earning growth as two independent stochastical variables, is equal to the summed variance of the two. 
#
# \begin{eqnarray}
# \overline{var^{r}}_{i,t} = \overline{var}_{i,t} + \overline{var}_{i,t}(\pi_{t})
# \end{eqnarray}
#
#
# Not enough information is available for the same kind of transformation of IQR and skewness from nominal to real, so I only use nominal variables. Besides, as there are extreme values on inflation expectations and uncertainty, I also exclude top and bottom $5\%$ of the observations. This further shrinks the sample, when using real moments, to 36,000. 
#

# #  Perceived income risks: basic facts 
#
#
# ##  Cross-sectional heterogeneity
#
# This section inspects some basic cross-sectional patterns of the subject moments of labor income. In the Figure \ref{fig:histmoms} below, I plot the histograms of $\overline\Delta y_{i,t}$, $\overline{var}_{i,t}$, $\overline {skw}_{i,t}$, $\overline {\Delta y^{r}}_{i,t}$, $\overline{var^{r}}_{i,t}$. 
#
# First, expected income growth across the population exhibits a dispersion ranging from a decrease of $2-3\%$ to around an increase of $10\%$ in nominal terms. Given the well-known downward wage rigidity, it is not surprising that most of the people expect a positive earning growth. At the same time, the distribution of expected income growth is right-skewed, meaning that more workers expect a smaller than larger wage growth. What is interesting is that this cross-sectional right-skewness in nominal earning disappears in expected real terms. Expected earnings growth adjusted by individual inflation expectation becomes symmetric around zero, ranging from a decrease of $10\%$ to an increase of $10\%$. Real labor income increase and decrease are approximately equally likely.  
#
# Second, as the primary focus of this paper, perceived income risks also have a notable cross-sectional dispersion. For both measures of risks variance and iqr, and in terms of both nominal and real terms, the distribution is right-skewed with a long tail. Specifically, most of the workers have perceived a variance of nominal earning growth ranging from zero to $20$ (a standard-deviation equivalence of $4-4.5\%$ income growth a year). But in the tail, some of the workers perceive risks to be as high as $7-8\%$ standard deviation a year. To have a better sense of how large the risk is, consider a median individual in our sample, who has an expected earning growth of $2.4\%$, and a perceived risk of $1\%$ standard deviation. This implies by no means negligible earning risk. 
#
# Third, the subjective skewness, an indicator of symmetry of the perceived density or upper/lower tail risk, are distributed across populations symmetrically around zero. It ranges from a left-skewness or negative skewness of 0.6 to the same size of positive skewness or right-skewness. Although one may think, based on the geenral knowledge of cross-sectional distribution of the earning growth,  a right-skewness is more common, it turns out that approximately equal proportion of the sample has left and right tails of their individual earning growth expectation. It is important to note here that this pattern is not particularly due to our density estimation assumptions. Both uniform and isosceles triangular distribution deliver a skewness of zero. (This is also why we can observe a clear cluster of the skewness at zero.) Therefore, the non-zero skewness estimates in our sample are both from the beta distribution cases, which is flexible enough to allow both. 
#
#

# + {"caption": "Distribution of Individual Moments", "code_folding": [], "label": "fig:histmoms", "note": "this figure plots histograms of the individual income moments. inc for nominal and rinc for real."}
## insert figures

graph_path = os.path.join(path,'../Graphs/ind/')

fig_list = ['hist_incexp.jpg',
            'hist_rincexp.jpg',
            'hist_incvar.jpg',
            'hist_rincvar.jpg']
            
nb_fig = len(fig_list)
    
file_list = [graph_path+ fig for fig in fig_list]

## show figures 
plt.figure(figsize=(10,10))
for i in range(nb_fig):
    plt.subplot(2,int(nb_fig/2),i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")
# -

# ##  Correlation with asset returns
#
# It is not only the labor income risk profile per se but also the macro risk profile, i.e. how the labor income is correlated with risky asset return and the business cycle, that is important for household decisions. Since the short time period of my sample (2013M6-2018M5) has not seen a single one business cycle, at least as defined by the NBER recession committee, it poses a challenge for me to examine the correlation between perceived risks and macroeconomic cycles. Therefore, as the first stage of the analysis, I only focus on the correlation between perceived risks and stock market returns. 
#
# Of course, there is a rationale in the first place to study stock market return and labor income, as it bears critical implications for household consumption insurance, equity premium, and participation puzzle. For instance, a negative correlation of income risk and risky asset return means households will be faced with higher risks of their total income by investing in the stock market. Or a negative correlation between skewness and stock market return, meaning a bigger income increase is less likely in low-return times will also deter households from participating in the stock market. 
#
# Following the most common practice in the finance literature, I use the monthly return of the S&P 500, computed from the beginning to the end of the month, as an approximate of the stock market return. Over the sample period, there are exactly two-thirds of the time marking a positive return. 
#
# For a population summary statistic of individual moments of perceived income growth, I take the median and mean across all respondents in the survey for each point of the time. One may worry about the seasonality of the monthly series of this kind. For instance, it is possible that workers tend to learn news about their future earnings at a particular month of the year, i.e. end of the fiscal year when the wage contracts are renegotiated and updated. Reasons of this kind may result in seasonal patterns of the expected earning growth, variance and other moments. Because my time series is too short in sample size to perform a trustful seasonal adjustment, I check the seasonality by inspecting the auto-correlation of each time series at different lags. As seen in the figures in the appendix, although it seems that the average or median earning growth per se has some seasonal patterns, there is no evidence for higher moments, such as variance and skewness. 
#
# There are two crucial econometric considerations when we examine the correlation between the subjective moments of earning growth and stock return.
#
# The first is the time-average or time-aggregation problem documented in both empirical asset pricing and consumption insurance literature (<cite data-cite="working_note_1960">(Working, 1960)</cite>, <cite data-cite="jagannathan_lazy_2007">(Jagannathan and Wang, 2007)</cite>, <cite data-cite="crawley_search_2019">(Crawley, 2019)</cite>).  Variables such as consumption and earning are interval measures, reported as an average over a period, while the stock return is a spot measure computed between two points of the time. As a result, if the unit of the time for the underlying income process is at a higher frequency than the measured interval (an extreme case being the continuous-time), the measured variable will exhibit upward biased autocorrelation and correlation with other underlying random walk series in the same frequency. In my context, such a problem can be partly mitigated by the availability of monthly frequency of earning expectations, if we assume the unit of time of the underlying stochastic process is a month. Then the directly observed monthly correlation of the two cannot be driven by the time aggregation problem. What also becomes immediately clear from this considration is that I should not examine the correlation of the two series in moving average terms, because it will cause the time aggregation problem. This point will be discussed in greater detail in the next section when I decompose the perceived income risks to different components of varying persistence. 
#
# The second issue regards which of the following, lagged, contemporaneous or forward is the correct correlation one should look at. Considering what is relevant to an individual making decisions are unrealized stochastic shocks to both income and asset return, one should examine the 1-year-ahead earning growth and its risks with the realized return over the impeding 12 months at each point of the time. 
#
# With these considerations, in the Figure \ref{fig:tssp500}, I plot the median perceived risk and skewness of both nominal and real earning along with the contemporaneous stock market returns by the end of each month (also true for the mean, see Figure \ref{micro_reg_exp} in appendix.). In order to account for the fact that the survey is undertaken in the middle of the month while the return is computed at the end of the month, I take the lag the income moments by 1 or 2 months when calculating the correlation coefficient. Table \ref{macro_corr} reports correlation coefficients of between perceived risks and the realized stock market return over the next 0-6 months. Although a Pearson test of the correlation coefficients is only significant for a 2-month lag, overall, the income risks measured by variance and IQR for both nominal and real earning post a negative correlation with the realized stock return a few months ahead. The subjective skewness has also a negative associated with the realized stock return in the near future. 
#
# More caution is needed when interpreting the observed negative association between perceived earning risks/skewness with stock market returns. First, my sample period is short and has mostly posted positive returns. Second, the pattern is based on a population median and mean of the perceived income risks, and it does not account for any household-specific characteristics. As we have seen in the cross-sectional pattern, there are substantial variations across individuals in their perceived income risks and skewness. Third, the risk profile we consider here is only relevant for marginal consumers/investors who at least have access to the stock market in the first place. Therefore, it is worth exploring the correlation above conditional on more individual characteristics. 
#  

# + {"caption": "Perceived Income Risks and Stock Market Return", "code_folding": [], "label": "fig:tssp500", "widefigure": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/pop/')

fig_list = ['tsMedmean.jpg',
            'tsMedvar.jpg',
            'tsEstMeanskew.jpg']
            
nb_fig = len(fig_list)

file_list = [graph_path+fig for fig in fig_list]



## show figures 

fig, ax = plt.subplots(figsize =(90,30),
                       nrows = nb_fig,
                       ncols = 1)
for i in range(nb_fig):
    ax[i].imshow(mpimg.imread(file_list[i]))
    ax[i].axis("off") 
plt.tight_layout(pad =0, w_pad=0, h_pad=0)

# + {"hide_input": true, "hide_output": true}
macro_corr  = pd.read_excel('../Tables/macro_corr.xlsx',index_col=0)
print('Correlation of perceived risks and stock return of x months ahead')
macro_corr
# -

#
#
# ##  Role of individual characteristics
#   
#    What factors are associated with subjective riskiness of labor income? This section inspects the question by regressing the perceived income risks at individual level on three major blocks of variables: job-specific characteristics, household demographics and other macroeconomic expectations held by the respondent. 
#
# In a general form, the regression is specified as followed, where the dependent variable is one of the individual subjetive moments that represent perceived income risks for either nominal or real earning. 
#
# \begin{eqnarray}
# \{\overline{var}_{i,t}, \overline{var}^r_{i,t}, \overline{iqr}_{i,t}\} = \alpha + \beta_0 \textrm{HH}_{i,t} + \beta_1 \textrm{JobType}_{i,t} + \beta_2 \textrm{Exp}_{i,t} + \beta_3 \textrm{Month}_t + \epsilon_{i,t}
# \end{eqnarray}
#
# The first block of factors, as called $\textit{Jobtype}_{i,t}$ includes dummy variables indicating if the job is part-time or if the work is for others or self-employed. Since the earning growth is specifically asked regarding the current job of the individual, I can directly test if a part-time job and the self-employed job is associated with higher perceived risks. 
#
# The second type of factors denoted $\textit{HH}_{i,t}$ represents household-specific demographics such as the household income level, education, and gender of the respondent. 
#
# Third, $\textit{Exp}_{i,t}$ represents other subjective expectations held by the same individual. As far as this paper is concerned, I include the perceived probability of unemployment herself, the probability of stock market rise over the next year and the probability of a higher nationwide unemployment rate. 
#
# $\textit{Month}_t$ is meant to control possible seasonal or month-of-the-year fixed effects. It may well be the case that at a certain point of the time of the year, workers are more likely to learn about news to their future earnings. But as I have shown in the previous section, such evidence is limited particularly for the higher moments of earnings growth expectations. 
#
# Besides, since many of the regressors are time-invariant household characteristics, I choose not to control household fixed effects in these regressions ($\omega_i$). Throughout all specifications, I cluster standard errors at the household level because of the concern of unobservable household heterogeneity. The regression results are presented in the Table \ref{micro_reg} below for three measures of perceived income risks, nominal growth variance, nominal growth iqr, and real growth variance. 
#
# The regression results are rather intuitive. It confirms that self-employed jobs, workers from low-income households and lower education have higher perceived income risks. In our sample, there are around $15\%$ (6000) of the individuals who report themselves to be self-employed instead of working for others. In the Table \ref{micro_reg_mean} shown in the appendix, this group of people also has higher expected earnings growth. The effects are statistically and economically significant. Whether a part-time job is associated with higher perceived risk is ambiguous depending on if we control household demographics. At first sight, part-time jobs may be thought of as more unstable. But the exact nature of part-time job varies across different types and populations. It is possible, for instance, that the part-time jobs available to high-income and educated workers bear lower risks than those by the low-income and low-education groups. 
#
# The negative correlation between perceived risks and household income is significant and robust throughout all specifications. In contrast, there is no such correlation between expected earning growth per se and household income. Although SCE asks the respondent to report an income range instead of the accurate monetary value, the 11-group breakdown is sufficiently granular to examine if the high-income/low risks association is monotonic. As implied by the size of the coefficient of each income group dummy in the Table \ref{micro_reg}, this pattern is monotonically negative until the top income group ($200k or above). I also plot the mean and median of income risks by income group in the Figure \ref{fig:boxplotbygroup}.  
#
# Besides household income, there is also statistical correlation between perceived risks and other demographic variables. In particular, higher eduation, being a male versus female, being a middle-aged worker compared to a young, are all associated with lower perceived income risks. To keep a sufficiently large sample size, I run regressions of this set of variables without controling the rest regressors.  Although the sample size shrink substantially by including these demographics, the relationships are statistically significant and consistent across all measures of earning risks. 
#
# Higher perceived the probability of losing the current job, which I call individual unemployment risk, $\textit{IndUE}$ is associated with higher earning risks of the current job. The perceived chance that the nationwide unemployment rate going up next year, which I call aggregate unemployment risk, $\textit{AggUE}$ has a similar correlation with perceived earning risks. Such a positive correlation is important because this implies that a more comprehensively measured income risk facing the individual that incorporates not only the current job's earning risks but also the risk of unemployment is actually higher. Moreover, the perceived risk is higher for those whose perceptions of the earning risk and unemployment risk are more correlated than those less correlated. 
#
# Lastly, what is ambiguous from the regression is the correlation between stock market expectations and perceived income risks. Although a more positive stock market expectation is associated with higher expected earnings growth in both real and nominal terms, it is positively correlated with nominal earning risks but negatively correlated with real earning risks. As the real earning risk is the summation of the perceived risk of nominal earning and inflation uncertainty, the sign difference has to be driven by a negative correlation of expectation stock market and inflation uncertainty.  In order to reach more conclusive statements, I will examine how perceived labor income risks correlate with the realized stock market returns and indicators of business cycles depending upon individual factors in the next step of analysis. 
#
# To summerize, a few questions arise from the patterns discussed above. First, what drives the differences in subjective earning risks across different workers? To what extent these perceptive differences reflect the true heterogeneity of the income risks facing by these individuals? Or they can be attributed to perceptive heterogeneity independent from the true risk profile. Second, how are individual earning risk is correlated with asset return expectations and broadly the macro economic environment? This will be the focus of the coming sections. 
#      
#     
#      

# + {"caption": "Perceived Income by Group", "code_folding": [], "label": "fig:boxplotbygroup", "note": "this figure is the boxplot of perceived income risk(inc for nominal and rinc for real) by different household income (HHinc), education (educ) and gender.", "widefigure": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/ind/')

fig_list = ['boxplot.jpg']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(50,50))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")
# -

reg_tb = pd.read_excel('../Tables/micro_reg.xlsx').replace(np.nan,'')

# + {"hide_input": true, "hide_output": true}
reg_tb
# -

# ##  Perceived income risks and decisions (in progress)
#
# This section investigates how individual-specific perceived risks are correlated with household economic decisions such as consumption and labor supply. I should note that the purpose of this exerice is not primarily for causal inference at current stage. Instead, it is meant to check if the surveyed households demostrate certain degree of in-survey consistency in terms of their perceptions and decision inclinations. 
#
# In particular, I ask two questions based on the available survey answers provided by the core module of the survey. First, are higher perceived income risks associated with a lower anticipated household spending growth? Second, are higher perceived income risks are associated with actions of self-insurance such as seeking an alternative job. This can be indirectly tested using the surveyed probability of voluntary seperation from the current job. In addition, supplementary modules of SCE have also surveyd more detailed questions on spending decisions and labor market. (These I will examine in the next stage of the analysis.)
#
# There is one important econometric concern when I run regressions of the decision variable on perceived risks due to the measurement error in the regressor used here. In a typical OLS regression in which the regressor has i.i.d. measurement errors, the coefficient estimate for the imperfectly measured regressor will have a bias toward zero. For this reason, if I find that willingness to consume is indeed negatively correlated with perceived risks, taking into acount the bias, it implies that the correlation of the two are greater in the magnitude. 
#
# The empirical results will be reported in the next version of the draft.

# #  Perceived income process (in progress) 
#
# The cross-sectional heterogeneity documented in the previous analysis in perceived risks mask two aspects that are crutial to understanding the real nature of the perceptual heterogeneity. First, to what extent these subjetive risk profiles simply reflect the de facto differences in the stochastic nature of the idiosyncratic labor income shocks? This is also imperative if one wants to attribute part of the differences to any deviations from full-information rational expectation. Second, do the perceived risks contain different components that differ in terms of its persistence? The risks associated with permanent and transitory shocks have substantially different impacts on individual decisions. In order to address both issues, I will need to specify a well-defined income process.
#
#
# ##  An illustration of the idea in a permanent-transitory income process
#
#
# The logged income of individual $i$ (excluding the predictable component given known information by agents at time $t$) follows a following process (same as <cite data-cite="carroll1997nature">(Carroll and Samwick, 1997)</cite>).  
#
# \begin{equation}
# \begin{split}
# y_{i,t} = p_{i,t} + \epsilon_{i,t} \\
# P_{i,t} = p_{i,t-1} + \theta_{i,t} \\
# \theta_{i,t} \sim N(0,\sigma_{\theta,t}) \\
# \epsilon_{i,t} \sim N(0,\sigma_{\epsilon,t})
# \end{split}
# \end{equation}
#
# where $p_{i,t}$ is a random walk component with a permanent shock $\theta_{i,t}$ in each point of time. $\epsilon_{i,t}$ is the transitory shock that is i.i.d.. The risks of both transitory and permanent shocks indicated by its variance are time-varying. For now, we do not break down individuals into different cohorts, i.e. $\sigma_{\theta,t}$ and $\sigma_{\epsilon,t}$ are not cohort specific. But we can do this exercise for any defined cohort.  
#
# Realized income growth is 
#
# \begin{equation}
# \begin{split}
# \Delta y_{i,t+1} = y_{i,t+1} - y_{i,t} \\
#  = P_{i,t+1} + \epsilon_{i,t+1} - P_{i,t} - \epsilon_{i,t} \\
#  = \theta_{i,t+1} + \Delta \epsilon_{i,t+1}
# \end{split}
# \end{equation}
#
# For an agent $i$, who knows perfectly about her income process, standing at time $t$, her conditional variance of income growth for next period is 
#
# \begin{equation}
#  Var^*_{i,t}(\Delta y_{i,t+1}) = \sigma^2_{\theta,t+1} + \sigma^2_{\epsilon,t+1} + \sigma^2_{\xi,t} \quad \forall i
# \end{equation}
#  
# where I use ${}^*$ supscript to denote the rational expecation version of the moments, in the sense that the individual $i$ knows perfectly her income process assumed here and have observed the realization of $\sigma_{\epsilon,t}$. Therefore the later does not show up in her condtional uncertainty. The third component $\sigma_{\xi,t}$ is an idiosyncratic shock to the level of perceived risk that can be either interpreted as a measurement error from the point view of the econometricians or a perceptual change to which econometricians have no access to. 
#
# At the same time, the cross-cetional variance of the expected income growth at time $t$ about income growth reflects the different views of the risks.
#
# \begin{equation}
# \overline {Var}^*_{t}(E_{i}(\Delta y_{i,t+1})) = \tilde \sigma^2_{\theta,t+1} +\tilde \sigma^2_{\epsilon,t}+ \tilde \sigma^2_{\epsilon,t+1}
# \end{equation}
#
#
# The autocovariance of expected income growth in consecutive two periods is as follows.
#
#
# \begin{equation}
# \overline {Cov}^*_{t+1|t}(E_{i,t}(\Delta y_{i,t+1}),E_{i,t+1}(\Delta y_{i,t+2}) ) = - \tilde \sigma^2_{\epsilon,t+1}
# \end{equation}
#
# The three moments exactly identify the perceived income risks in each period. One way to think about these risks is that they are revealed by people's forecasts.   
#
# These moments restrictions exactly mirrors the problem faced by econometricians who have only access to the realized earnings in a panel structure. 
#
# What is available to econometricians is the realized cross-sectional variance of income growth (no subscript $i$) shown below. It is different from uncertainty faced with individuals. 
#
# \begin{equation}
# Var (\Delta y_{i,t+1}) =  \sigma^2_{\theta,t+1} +\sigma^2_{\epsilon,t}+ \sigma^2_{\epsilon,t+1}
# \end{equation}
#
# Taking the differences of the population's analogue of the first equation and the second above recover variance of transitory risks $\sigma_{\epsilon,t}$. Recursively using the panel structure, we could recover all the transitory and permanent income risks.
#
# Besides, econometricians also use the following moments.
#
# \begin{equation}
# Cov (\Delta y_{i,t}, \Delta y_{i,t+1}) =  -\sigma^2_{\epsilon,t}
# \end{equation}
#
# This exercise is based on the assumption that individuals across the population or one defined cohort share the same income process. And also it is rational expectation in the sense that on average individuals get the income process right. 
#
# Once we recover permanent and transitory volatilities from above exercise, we can compare them with estimates from only realized income serieses.   
#
# ##  Other moments from rational expectation
#
# Besises, econometricians have utilized another moment restrictions: auto correlation of income growth across two periods are 
# \begin{equation}
# Cov^*_{t}( \Delta y_t, \Delta y_{t+1} ) = \\
#  = Cov^*_{t}(\theta_t + \epsilon_t - \epsilon_{t-1}, \theta_{t+1} + \epsilon_{t+1} - \epsilon_{t}) \\
#  = 0 
# \end{equation}
#
# This is, again, different to an econometrician, for whom the covariance is $-\sigma^2_{\epsilon,t}$. The rational agent in the model learns about $\sigma_{\epsilon,t}$.  
#
# The serial covariance of expeced income growth across two periods are 
# \begin{equation}
# Cov^*( E_{t-1}(\Delta y_t), E_t(\Delta y_{t+1}) ) = \\
# = Cov^*(E_{t-1}(\theta_t +\epsilon_t - \epsilon_{t-1}), E_{t}(\theta_{t+1} + \epsilon_{t+1} - \epsilon_t)) \\
# = 0
# \end{equation}

# ## Time aggregation problem 
#
# - The earning growth asked is from $m$ to $m+12$. 
# - The survey is asked each month. 

# ###  A simple example with half-year as the unit of the time 
#
# Earning in year $t$ is a summation of half-year earning. 
#
# \begin{equation}
# y_t = y_{t_2}+ y_{t_2} 
# \end{equation}
#
# The YoY growth of income is below
#
# \begin{equation}
# \begin{split}
# \Delta y_{t_2+1} = y_{(t+1)_1}+ y_{(t+1)_2} - y_{t_1 } - y_{t_2}  \\
#  = p_{(t+1)_1} + \epsilon_{(t+1)_2} + p_{(t+1)_2} + \epsilon_{(t+1)_2} - p_{t_1} - \epsilon_{t_1} - p_{t_1} - \epsilon_{(t)_2 } \\
#  = \theta_{(t)_2} + \theta_{(t+1)_1} + \theta_{(t+1)_2} + \theta_{(t+1)_1} + \epsilon_{(t+1)_1} + \epsilon_{(t+1)_2} - \epsilon_{t_1} - \epsilon_{t_2} \\
#  =  \theta_{t_2} + 2\theta_{(t+1)_1} + \theta_{(t+1)_2} + \epsilon_{(t+1)_1} + \epsilon_{(t+1)_2} - \epsilon_{t_1} - \epsilon_{t_2} 
# \end{split}
# \end{equation}
#
# The middle-year-on-middle-year income growth is
#
#
# \begin{equation}
# \begin{split}
# \Delta y_{(t+1)_1+1} = y_{(t+1)_2}+ y_{(t+2)_1} - y_{(t+1)_1} - y_{t_2}  \\
#  = p_{(t+1)_2} + \epsilon_{(t+1)_2} + p_{(t+2)_1} + \epsilon_{(t+2)_1} - p_{(t+1)_1} - \epsilon_{(t+1)_1} - p_{t_2} - \epsilon_{t_2 } \\
#  = \theta_{(t+1)_2} + \theta_{(t+1)_1} + \theta_{(t+1)_2} + \theta_{(t+2)_1} + \epsilon_{(t+1)_2} + \epsilon_{(t+2)_1} - \epsilon_{(t+1)_1} - \epsilon_{t_2 } \\
#  = 2\theta_{(t+1)_2} + \theta_{(t+1)_1} + \theta_{(t+2)_1} + \epsilon_{(t+1)_2} + \epsilon_{(t+2)_1} - \epsilon_{(t+1)_1} - \epsilon_{t_2 }
# \end{split}
# \end{equation}
#
#
# Then for each individual $i$ at $t''$ and $(t+1)'$ are respectively: 
#
# \begin{equation}
# Var^*_{i,t_2}(\Delta y_{i,t_2+1}) =  2\sigma^2_{\theta,(t+1)_1} + \sigma^2_{\theta,(t+1)_2} + \sigma^2_{\epsilon,(t+1)_1} + \sigma^2_{\epsilon,(t+1)_2}
# \end{equation}
#
#
# \begin{equation}
# Var^*_{i,(t+1)_1}(\Delta y_{i,(t+1)_1+1}) =  2\sigma^2_{\theta,(t+1)_2} + \sigma^2_{\theta,(t+2)_1} + \sigma^2_{\epsilon,(t+1)_2} + \sigma^2_{\epsilon,(t+2)_1}
# \end{equation}
#
# From end of $t_2$ (end of year $t$) to the end of $(t+1)_1$ (middle of the year $t+1$), the realization of $\theta_{(t+1)_1}$ and $\epsilon_{(t+1)_1}$ reduces the variance. 
#
#
# Besides, the econometricians have access to following two cross-sectional moments.
#
# \begin{equation}
# Var (\Delta y_{i,t_2+1}) =  \sigma^2_{\theta,t_2} + 2\sigma^2_{\theta,(t+1)_1} + \sigma^2_{\theta,(t+1)_2} + \sigma^2_{\epsilon,(t+1)_1} + \sigma^2_{\epsilon,(t+1)_2} + \sigma^2_{\epsilon,t_1} + \sigma^2_{\epsilon,t_2} 
# \end{equation}
#
#
# \begin{equation}
# Var (\Delta y_{i,(t+1)_1+1}) =  2\sigma^2_{\theta,(t+1)_2} + \sigma^2_{\theta,(t+1)_1} + \sigma^2_{\theta,(t+2)_1} + \sigma^2_{\epsilon,(t+1)_2} + \sigma^2_{\epsilon,(t+2)_1} + \sigma^2_{\epsilon,(t+1)_1} + \sigma^2_{\epsilon,t_2}
# \end{equation}
#
# \begin{equation}
# \begin{split}
# Cov ( \Delta y_{i,(t-1)_2+1},\Delta y_{i,t_1+1}) = Cov(\theta_{(t-1)_2} + 2\theta_{t_1} + \theta_{t_2} + \epsilon_{t_1} + \epsilon_{t_2} - \epsilon_{(t-1)_1} - \epsilon_{(t-1)_2} , \\
# 2\theta_{t_2} + \theta_{t_1} + \theta_{(t+1)_1} + \epsilon_{t_2} + \epsilon_{(t+1)_1} - \epsilon_{t_1} - \epsilon_{(t-1)_2 } ) \\
# = 2\sigma^2_{\theta,t_1} + 2\sigma^2_{\theta,t_2} - \sigma^2_{\epsilon,t_1} + \sigma^2_{\epsilon,t_2} + \sigma^2_{\epsilon,(t-1)_2}
# \end{split}
# \end{equation}
#
# \begin{equation}
# \begin{split}
# Cov ( \Delta y_{i,(t-1)_2+1},\Delta y_{i,t_2+1}) = Cov(\theta_{(t-1)_2} + 2\theta_{t_1} + \theta_{t_2} + \epsilon_{t_1} + \epsilon_{t_2} - \epsilon_{(t-1)_1} - \epsilon_{(t-1)_2} , \\
# \theta_{t_2} + 2\theta_{(t+1)_1} + \theta_{(t+1)_2} + \epsilon_{(t+1)_1} + \epsilon_{(t+1)_2} - \epsilon_{t_1} - \epsilon_{t_2} ) \\
# = \sigma^2_{\theta,t_2}-(\sigma^2_{\epsilon,(t+1)_1} + \sigma^2_{\epsilon,t_2})
# \end{split}
# \end{equation}
#
# \begin{equation}
# \begin{split}
# Cov ( \Delta y_{i,t_2+1},\Delta y_{i,(t+1)_2}) = \sigma^2_{\theta,(t+1)_2}-(\sigma^2_{\epsilon,(t+2)_1} + \sigma^2_{\epsilon,(t+1)_2})
# \end{split}
# \end{equation}
#
#
# The rational expectation assumption also gives following moment restrictions
#
# \begin{equation}
# Cov^*_{t_2}(\Delta y_t, \Delta y_{t+1}) = 0
# \end{equation}
#
# Standing at any point of the time, for the rational agent, the $\Delta y_t$ is realizated already. So it should have zero covariance with income growth in future. 
#
# This is again, different from the econometrician's problem. 
#
#

# #  Model (in progress)

#
# #  Summary 
#

# + {"hide_cell": true, "cell_type": "markdown"}
# # Appendix 
