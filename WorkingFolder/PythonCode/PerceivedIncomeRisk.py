# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Introduction
#
#
# Income risks matter for both individual behaviors and aggregate outcomes. With identical expected income and homogeneous risk preferences, different degrees of risks lead to different saving/consumption and portfolio choices. This is well understood in models in which agents are inter-temporally risk-averse, or prudent (<cite data-cite="kimball1990precautionary">(Kimball, 1990)</cite>, <cite data-cite="carroll2001liquidity">(Carroll and Kimball, 2001)</cite>), and the risks associated with future marginal utility motivate precautionary motives. Since it is widely known from the empirical research that idiosyncratic income risks are at most partially insured (<cite data-cite="blundell_consumption_2008">(Blundell, et al., 2008)</cite>) or because of borrowing constraints, such behavioral regularities equipped with market incompleteness leads to ex-post unequal wealth distribution and different degrees of marginal propensity to consume (MPC) (<cite data-cite="huggett1993risk">(Huggett,1993)</cite>, <cite data-cite="aiyagari1994uninsured">(Aiyagari, 1994)</cite>). This has important implications for the transmission of macroeconomic policies\footnote{<cite data-cite="krueger2016macroeconomics">(Krueger, et al., 2016)</cite>, <cite data-cite="kaplan2018monetary">(Kaplan, et al., 2018)</cite>, <cite data-cite="auclert2019monetary">(Auclert, 2019)</cite>, <cite data-cite="bayer2019precautionary">(Bayer, et al., 2019)</cite>.}   
#
# One important assumption prevailing in macroeconomic models with uninsured risks is that agents have a perfect understanding of the income risks. Under the assumption, economists typically estimate the income process based on micro income data and then treat the estimates as the true model parameters known by the agents making decisions in the model\footnote{For example, <cite data-cite="krueger2016macroeconomics">(Krueger, et al.,2016)</cite>, <cite data-cite="bayer2019precautionary">(Bayer, et al., 2019)</cite>.}. But given the mounting evidence that people form expectations in ways deviating from full-information rationality, leading to perennial heterogeneity in economic expectations held by micro agents, this assumption seems to be too stringent. To the extent that agents make decisions based on their *respective* perceptions, understanding the *perceived* income risk profile and its correlation structure with other macro variables are the keys to explaining their behavior patterns.
#
# The theoretical contribution of this paper on this front is to establish a unified framework for perceived income risks under different possible income processes seen in the macro literature. Under a clearly specified income process, I can examine to what extent the perceived income risks align with a number of benchmark predictions under full-information rational expectation(FIRE) and with a list of empirically documented facts regarding the income risk dynamics. For instance, is there a large degree of dispersion in risk perceptions among agents who the modelers assume face the same level of risks? Or are perceived risks state-dependent and countercyclical as documented by some literature? Does the perceived risk reflect a quasi-perfect understanding of the income risks of different nature? 
#
# Individuals of varying characteristics face potentially different income processes. Even under the same income process, the realizations of income differ across agents due to differences in realized shocks. In addition to the fact that realized income is not observed in these surveys, this makes it additionally challenging to undertake comparisons between perceptions and the underlying process in a similar manner as for expectations about macroeconomic variables such as inflation. A clear comparison of such spirit is also possibly sensitive to the consistency between the frequency of the reported income perception and the frequency of the underlying income process,  i.e. the time aggregation problem. Besides, I also explicitly take into account the presence of the superior information problem extensively discussed in the literature. 
#
# After clarifying these issues in the theory, I proceed to establish the empirical facts regarding income risk perceptions. Spefically, I utilize the recently available density forecasts of labor income surveyed by New York Fed's Survey of Consumer Expectation (SCE). What is special about this survey is that agents are asked to provide histogram-type forecasts of their earning growth over the next 12 months together with a set of expectational questions about the macroeconomy. When the individual density forecast is available, a parametric density estimation can be made to obtain the individual-specific subjective distribution. And higher moments reflecting the perceived income risks such as variance, as well as the asymmetry of the distribution such as skewness allow me to directly characterize the perceived risk profile without relying on external estimates from cross-sectional microdata. This provides the first-hand measured perceptions on income risks that are truly relevant to individual decisions.
#
# Perceived income risks exhibits a number of important patterns that are consistent with the predictions of my model of experience-based learning with subjective attribution. 
#
# - Higher experienced volatility is associated with higher perceived income risks. This helps explain why perceived risks differ systematically across different generations, who have experienced different histories of the income shocks. Besides, perceived risks declines with one's age.
#
# - Perceived income risks have a non-monotonic correlation with the current income, which can be best described as a skewed U shape. Perceived risk decreases with current income over the most range of income values follwed by an uppick in perceived risks for high-income group. 
#  
# - Perceived income risks are counter-cyclical with the labor market conditions or broadly business cycles. I found that average perceived income risks by U.S. earners are negatively correlated with the current labor market tightness measured by wage growth and unemployment rate. Besides, earners in states with higher unemployment rates and low wage growth also perceive income risks to be higher. This bears similarities to but important difference with a few previous studies that document the counter-cyclicality of income risks estimated by cross-sectional microdata (<cite data-cite="guvenen2014nature">(Guvenen, et al., 2014)</cite>, <cite data-cite="catherine_countercyclical_2019">(Catherine, 2019)</cite>). 
#
# - Perceived income risks translate into economic decisions in a way consistent with precautionary saving motives. In particular, households with higher income risk perceptions expect a higher growth in expenditure, i.e. lower consumption today versus tomorrow.  
#
#
# These patterns suggest that individuals have a roughly good yet imperfect understanding of their income risks. Good, in the sense that subjective perceptions are broadly consistent with the realization of cross-sectional income patterns. This is attained in my model because agents learn from past experiences, roughly as econometricians do. In contrast, subjective perceptions are imperfect in that bounded rationality prevents people from knowing about the true size and nature of income shocks as well some parameters of the process perfectly. If hardworking economists equipped with advanced econometrical techniques and a large sample of income data do not necessarily specify the income process correctly, it is feasible to admit the agents in the model to be subject to the same difficulty. 
#
# As illustrated by much empirical work of testing the rationality in expectations, it is admittedly challenging to separately account for the differences in perceptions driven by the "truth" and the part driven by the pure subjective heterogeneity. The most straightforward way seems to be to treat econometrician's external estimates of the income process as the proxy to the truth, for which the subjective surveys are compared. But this approach implicitly assumes that econometricians correctly specify the model of the income process and ignores the possible superior information that is available only to the people in the sample but not to econometricians. The model built in this paper reconciles both possibilities by modeling agents as boundedly rational econometricians subject to model misspecifications. 
#  
# Finally, the subjective learning model will be incorporated into an otherwise standard life-cycle consumption/saving model with uninsured idiosyncratic and aggregate risks. Experience-based learning makes income expectations and risks state-dependent when agents make dynamically optimal decisions at each point of the time. In particular, higher perceived risks will induce more precautionary saving behaviors. If this perceived risk is state-dependent on recent income changes, it will potentially shift the distribution of MPCs along income deciles, therefore, amplify the channels aggregate demand responses to shocks. 
#
#      
# ##  Related literature
#
# This paper is the closest to the literature on income risks, precautionary saving, and the partial insurance. 
#
# - <cite data-cite="gottschalk1994growth">(Gottschalk et al. 1994)</cite>
# - <cite data-cite="carroll1997nature">(Carroll and Samwick, 1997)</cite>
# - <cite data-cite="meghir2004income">(Storesletten et al. 2004)</cite>, 
# - <cite data-cite="storesletten2004cyclical">(Meghir and Pistaferri, 2004)</cite>, 
# - <cite data-cite="blundell_consumption_2008">(Blundell et al. 2008)</cite>
# - <cite data-cite="guvenen2014nature">(Guvenen et al. 2014)</cite>
# - <cite data-cite="bloom2018great">(Bloom et al. 2018)</cite>
#
#
#
# Besides, this paper is relevant to four lines of literature. First, it is related to an old but recently reviving interest in studying consumption/saving behaviors in models incorporating imperfect expectations and perceptions. For instance, the closest to the current paper, <cite data-cite="pischke1995individual">(Pischke, 1995)</cite> explores the implications of the incomplete information about aggregate/individual income innovations by modeling agent's learning about inome component as a signal extraction problem. <cite data-cite="wang2004precautionary">(Wang, 2004)</cite> extends the framework to incorporate precautionary saving motives. In a similar spirit, <cite data-cite="carroll_sticky_2018">(Carroll et al. 2018)</cite> reconciles the low micro-MPC and high macro-MPCs by introducing to the model an information rigidity of households in learning about macro news while being updated about micro news. <cite data-cite="rozsypal_overpersistence_2017">(Rozsypal and Schlafmann, 2017)</cite> found that households' expectation of income exhibits an over-persistent bias using both expected and realized household income from Michigan household survey. The paper also shows that incorporating such bias affects the aggregate consumption function by distorting the cross-sectional distributions of marginal propensity to consume(MPCs) across the population.  <cite data-cite="lian2019imperfect">(Lian, 2019)</cite> shows that an imperfect perception of wealth accounts for such phenomenon as excess sensitivity to current income and higher MPCs out of wealth than current income and so forth. My paper has a similar flavor to all of these works in that I also explore the behavioral implications of households' perceptual imperfection. But it has important two distinctions. First, this paper focuses on higher moments such as income risks. Second, most of these existing work either considers inattention of shocks or bias introduced by the model parameter, none of these explores the possible misperception of the nature of income shocks. \footnote{For instance, <cite data-cite="pischke1995individual">(Pischke, 1995)</cite> assumes that agents know perfectly about the variance of permanent and transitory income so that they could filter the two components from observable income changes. This paper instead assumes that that the agents do not observe the two perfectly.} 
#
#
# Second, empirically, this paper also contributes to the literature studying expectation formation using subjective surveys. There has been a long list of "irrational expectation" theories developed in recent decades on how agents deviate from full-information rationality benchmark, such as sticky expectation, noisy signal extraction, least-square learning, etc. Also, empirical work has been devoted to testing these theories in a comparable manner (<cite data-cite="coibion2012can">(Coibion and Gorodnichenko, 2012)</cite>, <cite data-cite="fuhrer2018intrinsic">(Fuhrer, 2018)</cite>). But it is fair to say that thus far, relatively little work has been done on individual variables such as labor income, which may well be more relevant to individual economic decisions. Therefore, understanding expectation formation of the individual variables, in particular, concerning both mean and higher moments, will provide fruitful insights for macroeconomic modeling assumptions. 
#
# Third, the paper is indirectly related to the research that advocated for eliciting probabilistic questions measuring subjective uncertainty in economic surveys (<cite data-cite="manski_measuring_2004">(Manski, 2004)</cite>, <cite data-cite="delavande2011measuring">(Delavande et al. 2011)</cite>, <cite data-cite="manski_survey_2018">(Manski, 2018)</cite>). Although the initial suspicion concerning to people’s ability in understanding, using and answering probabilistic questions is understandable, <cite data-cite="bertrand_people_2001">(Bertrand and Mullainathan,2001)</cite> and other works have shown respondents have the consistent ability and willingness to assign a probability (or “percent chance”) to future events. <cite data-cite="armantier_overview_2017">(Armantier et al. 2017)</cite>  have a thorough discussion on designing, experimenting and implementing the consumer expectation surveys to ensure the quality of the responses. Broadly speaking, the advocates have argued that going beyond the revealed preference approach, availability to survey data provides economists with direct information on agents’ expectations and helps avoids imposing arbitrary assumptions. This insight holds for not only point forecast but also and even more importantly, for uncertainty, because for any economic decision made by a risk-averse agent, not only the expectation but also the perceived risks matter a great deal.
#
# Lastly, the idea of this paper echoes with an old problem in the consumption insurance literature: 'insurance or information' (<cite data-cite="pistaferri_superior_2001">Pistaferri, 2001</cite>, <cite data-cite="kaufmann_disentangling_2009">Kaufmann and Pistaferri, 2009</cite>,<cite data-cite="meghir2011earnings">Meghir et al. 2011</cite>). In any empirical tests of consumption insurance or consumption response to income, there is always a worry that what is interpreted as the shock has actually already entered the agents' information set or exactly the opposite. For instance, the notion of excessive sensitivity, namely households consumption highly responsive to anticipated income shock, maybe simply because agents have not incorporated the recently realized shocks that econometricians assume so (<cite data-cite="flavin_excess_1988">Flavin,1988</cite>). Also, recently, in the New York Fed [blog](https://libertystreeteconomics.newyorkfed.org/2017/11/understanding-permanent-and-temporary-income-shocks.html), the authors followed a similar approach to decompose the permanent and transitory shocks. My paper shares a similar spirit with these studies in the sense that I try to tackle the identification problem in the same approach: directly using the expectation data and explicitly controlling what are truly conditional expectations of the agents making the decision. This helps economists avoid making assumptions on what is exactly in the agents' information set. What differentiates my work from other authors is that I focus on higher moments, i.e. income risks and skewness by utilizing the recently available density forecasts of labor income. Previous work only focuses on the sizes of the realized shocks and estimates the variance of the shocks using cross-sectional distribution, while my paper directly studies the individual specific variance of these shocks perceived by different individuals.  
#

# + {"code_folding": [0], "hide_output": true}
## import libraries for inserting figures 

import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline 
from IPython.display import display, Image
import matplotlib.image as mpimg
import os
import pandas as pd

path = os.getcwd()

"""
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
    
"""
# -

# # Theoretical framework 
#
# ## Income process and risk perceptions
#
# Log income of individual $i$ from cohort $c$ at time $t$ follows the following process (<cite data-cite="meghir2004income">(Meghir and Pistaferri(2004)</cite>). Cohort $c$ represents the year of entering the job market. It contains a predictable component $z$, and a stochastical component $e$. The latter consists of aggregate component $g$, idiosyncratic permanent $\psi$, MA(1) component $\eta$ and a transitory component $\theta$. Here I do not consider unemployment risk since the perceived risk measured in the survey conditions on staying employed.
#
# \begin{equation}
# \begin{split}
# & y_{i,c,t} = z_{i,c,t}+e_{i,c,t} \\
# & e_{i,c,t}=g_t+p_{i,c,t}+\eta_{i,c,t}+\theta_{i,c,t}  \\
# & g_{t+1} = g_t + \xi_{t+1} \\
# & p_{i,c,t+1} = p_{i,c,t}+\psi_{i,c,t-1} \\
# & \eta_{i,c,t+1} = \phi\epsilon_{i,c,t}+\epsilon_{i,c,t+1}
# \end{split}
# \end{equation}
#
# All shocks including the aggregate one, and idiosyncratic ones follow normal distributions with zero means and time-invariant variances denoted as $\sigma^2_{\xi}$, $\sigma^2_{\theta}$,$\sigma^2_{\epsilon}$,$\sigma^2_{\psi}$. Hypothetically, these variances could differ both across cohorts and time. I focus on the most simple case here and results with cohort-specific income risks are reported in the Appendix. 
#
# Income growth from $t$ to $t+1$ consists predictable changes in $z_{i,c,t+1}$, and those from realized income shocks. 
#
# \begin{equation}
# \begin{split}
# \Delta y_{i,c,t+1} & =  \Delta z_{i,c,t+1}+\Delta e_{i,c,t} \\
# &= \Delta z_{i,c,t+1}+\xi_{t+1}+\psi_{i,c,t+1}+\epsilon_{i,c,t+1}+(\phi-1)\epsilon_{i,c,t}-\phi\epsilon_{i,c,t-1}+\theta_{i,c,t+1}
# \end{split}
# \end{equation}
#
#
# All shocks that have realized till $t$ are observed by the agent at time $t$. Therefore, under full-information rational expectation(FIRE), namely when the agent perfectly knows the income process and parameters, the perceived income risks or the perceived variance of income growth from $t$ to $t+1$ is equal to the following.
#
# \begin{equation}
# Var_{t}^*(\Delta y_{i,c,t+1}) =Var_{t}^*(\Delta e_{i,c,t+1}) =   \sigma^2_\xi+\sigma^2_\psi + \sigma^2_\epsilon+\sigma^2_\theta 
# \end{equation}
#
# FIRE has a number of testable predictions about the behaviors of perceived risks. 
#
# - First, agents who share the same income process have no disagreements on perceived risks. This can be checked by comparing within-cohort/group dispersion in perceived risks. 
#
# - Second, the perceived risks under such an assumed income process are not dependent on past/recent income realizations. This can be tested by estimating the correlation between perceived risks and past income realizations or their proxies if the latter is not directly observed. 
#
# - Third, under the assumed progress, the variances of different-natured shocks sum up to exactly the perceived income risks and the loadings of all components are all positive. I report detailed derivations and proofs in the Appendix that these predictions are robust to the alternative income process and the time-aggregation problem discussed in the literature. The latter arises when the underlying income process is set at a higher frequency than the observed frequency of reported income or income expectations. This will cause different sizes of loadings of all future shocks to perceived annual risk but does not change the positivity of the loadings from different components onto perceived risk. 
#
# The challenge of testing the third prediction is that the risk parameters are not directly observable. Econometricians and modelers usually estimate them relying upon cross-sectional moment information from some panel data and take them as the model parameters understood perfectly by the agents. I can therefore use econometricians' best estimates using past income data as the FIRE benchmark (I will discuss the concerns of this approach later).
# Assuming the unexplained income residuals from this estimation regression is $\hat e_{i,t}= y_{i,c,t}-\hat z_{i,c,t}$($\hat z_{i,c,t}$ is the observable counterpart of $z_{i,c,t}$ from data). The unconditional cross-sectional variance of the change in residuals(equivalent to the ``income volatility'' or ``instability'' in the literature) is the following. Let's call this growth volatility. It can be further decomposed into different components in order to get the component-specific risk. 
#
# \begin{equation}
# Var(\Delta \hat e_{i,c,t}) = \hat\sigma^2_\xi+\hat\sigma^2_\psi + ((1-\phi)^2+\phi^2+1)\hat\sigma^2_\epsilon+\hat\sigma^2_\theta 
# \end{equation}
#
# Notice the unconditional growth volatility overlaps with the FIRE perceived risk in every component of the risks. But it is unambiguously greater than the perceived risk under FIRE because econometricians' do not directly observe the MA(1) shock from $t-1$. But this suffices to suggest that growth volatility is positively correlated with perceived risk. 
#
# Corresponding to the growth volatility, let's also define the level volatility as the cross-sectional variance of the levels of the residuals, which is denoted by $Var(\hat e_{i,t})$. Different from growth volatility, it includes the cumulative volatility from all the past permanent shocks as well as the MA shock from $t-1$, all of which are not correlated with the perceived risk under FIRE. Therefore, it any it will have only a weak correlation with perceived risks under FIRE. 
#
# This suggests the fourth testable prediction stated as below. Since we can obtain $Var(\Delta \hat e_{i,c,t})$ and $Var(\hat e_{i,c,t})$ using past income data, we can test this prediction.   
#
# - Growth volatility and risk perceptions are positively correlated. In contrast, level volatility is only weakly correlated with perceived risks. 
#
# It is worth asking how sensitive this prediction is to possible model misspecification of the income process in the first place. We consider three most common issues in the literature regarding the income risks. 
#
# - __Permanent versus persistent shock__. Replacing the permanent shock to $p$ with a persistent one in the above process essentially adds another component from the previous period to the growth in residuals, hence increases overall unconditional volatility. In the meantime, it does not change the perceived risk under FIRE. Therefore, the effect of making the permanent shock persistent will lead to a smaller correlation between FIRE risk perception and growth volatility. But the correlation will remain positive. 
#
# - __Moving average shock or purely transitory shock__. Our model actually nests both cases. Setting $\phi=0$ corresponds to purely transitory shocks. Any positive $\phi$ allows the coexistence of MA shock and transitory shocks. Our test is also robust to this alternative specification. 
#
# - __Time-invariant versus time-varying volatility__. Under the former assumption, the income volaility estimated from past income data can be directly comparable with the perceived risks reported for a different period. But doing so under the later assumption is inconsistent with the model. It requires the perceived risks and the realized income data are for the same time horizon. This is hard to satisfy based on the current data avaiability. But what's assuring is that if the stochastical volatilities are persistent over the time, we should still expect to see positive correlation between past volatility and FIRE risk perception even if the former is estimated from an earlier period.  
#
# There is another complication regarding the FIRE test: the superior information problem. It states that what econometrician's treat as income shocks are actually in the information set of the FIRE agents. Think this as when the known characteristics $\hat z$ used in the regression only partially captures the true predictable components $z$. Hence the sample residuals $\hat e$ are bigger than its true counterparts and this results in higher estimated growth and level volatility from data than the level relevant to FIRE agents in the model. It is true that this leads to a lower correlation between volatility and perceived risks, but it does not alter the prediction about the positive correlation between the two.

# # Data, variables and density estimation
#
# ## Data
#
# The data used for this paper is from the core module of Survey of Consumer Expectation(SCE) conducted by the New York Fed, a monthly online survey for a rotating panel of around 1,300 household heads over the period from June 2013 to January 2020, over a total of 80 months. This makes about 95113 household-year observations, among which around 68361 observations provide non-empty answers to the density question on earning growth. 
#
# Particularly relevant for my purpose, the questionnaire asks each respondent to fill perceived probabilities of their same-job-hour earning growth to pre-defined non-overlapping bins. The question is framed as "suppose that 12 months from now, you are working in the exact same [“main” if Q11>1] job at the same place you currently work and working the exact same number of hours. In your view, what would you say is the percentage chance that 12 months from now: increased by x% or more?".
#
# As a special feature of the online questionnaire, the survey only moves on to the next question if the probabilities filled in all bins add up to one. This ensures the basic probabilistic consistency of the answers crucial for any further analysis. Besides, the earning growth expectation is formed for exactly the same position, same hours, and the same location. This has two important implications for my analysis. First, these conditions help make sure the comparability of the answers across time and also excludes the potential changes in earnings driven by endogenous labor supply decisions, i.e. working for longer hours. Empirical work estimating income risks are often based on data from received income in which voluntary labor supply changes are inevitably included. Our subjective measure is not subject to this problem and this is a great advantage. Second, the earning expectations and risks measured here are only conditional on non-separation from the current job. It excludes either unemployment, i.e. likely a zero earning, or an upward movement in the job ladder, i.e. a different earning growth rate. Therefore, this does not fully reflect the entire income risk profile relevant to each individual. 
#
# Unemployment and other involuntary job separations are undoubtedly important sources of income risks, but I choose to focus on the same-job/hour earning with the recognition that individuals' income expectations, if any, may be easier to be formed for the current job/hour than when taking into account unemployment risks. Given the focus of this paper being subjective perceptions, this serves as a  useful benchmark.  What is more assuring is that the bias due to omission of unemployment risk is unambiguous. We could interpret the moments of same-job-hour earning growth as an upper bound for the level of growth rate and a lower bound for the income risk. To put it in another way, the expected earning growth conditional on current employment is higher than the unconditional one, and the conditional earning risk is lower than the unconditional one. At the same time, since SCE separately elicits the perceived probability of losing the current job for each respondent, I could adjust the measured labor income moments taking into account the unemployment risk. 
#
# ## Density estimation and variables 
#
# With the histogram answers for each individual in hand, I follow <cite data-cite="engelberg_comparing_2009">(Engelberg, Manskiw and Williams, 2009)</cite> to fit each of them with a parametric distribution accordingly for three following cases. In the first case when there are three or more intervals filled with positive probabilities, it was fitted with a generalized beta distribution. In particular, if there is no open-ended bin on the left or right, then two-parameter beta distribution is sufficient. If there is either open-ended bin with positive probability, since the lower bound or upper bound of the support needs to be determined, a four-parameter beta distribution is estimated. In the second case, in which there are exactly two adjacent intervals with positive probabilities, it is fitted with an isosceles triangular distribution. In the third case, if there is only one positive-probability of interval only, i.e. equal to one, it is fitted with a uniform distribution. 
#
# Since subjective moments such as variance is calculated based on the estimated distribution, it is important to make sure the estimation assumptions of the density distribution do not mechanically distort my cross-sectional patterns of the estimated moments. This is the most obviously seen in the tail risk measure, skewness. The assumption of log normality of income process, common in the literature (See again <cite data-cite="blundell_consumption_2008">(Blundell et al. 2008)</cite>), implicitly assume zero skewness, i.e. that the income increase and decrease from its mean are equally likely. This may not be the case in our surveyed density for many individuals. In order to account for this possibility, the assumed density distribution should be flexible enough to allow for different shapes of subjective distribution. Beta distribution fits this purpose well. Of course, in the case of uniform and isosceles triangular distribution, the skewness is zero by default. 
#
# Since the microdata provided in the SCE website already includes the estimated mean, variance and IQR by the staff economists following the exact same approach, I directly use their estimates for these moments. At the same time, for the measure of tail-risk, i.e. skewness, as not provided, I use my own estimates. I also confirm that my estimates and theirs for the first two moments are correlated with a coefficient of 0.9. 
#
# For all the moment's estimates, there are inevitably extreme values. This could be due to the idiosyncratic answers provided by the original respondent, or some non-convergence of the numerical estimation program. Therefore, for each moment of the analysis, I exclude top and bottom $3\%$ observations, leading to a sample size of around 48,000. 
#
# I also recognize what is really relevant to many economic decisions such as consumption is real income instead of nominal income. I, therefore, use the inflation expectation and inflation uncertainty (also estimated from density question) to convert nominal earning growth moments to real terms for some robustness checks in this paper. In particular, the real earning growth rate is expected nominal growth minus inflation expectation. 
#
#
# \begin{eqnarray}
# \overline {\Delta y^{r}}_{i,t} = \overline\Delta y_{i,t} - \overline \pi_{i,t}
# \end{eqnarray}
#
# The variance associated with real earning growth, if we treat inflation and nominal earning growth as two independent stochastic variables, is equal to the summed variance of the two. The independence assumption is admittedly an imperfect assumption because of the correlation of wage growth and inflation at the macro level. So it is should be interpreted with caution. 
#
# \begin{eqnarray}
# \overline{var}_{i,t}(\Delta y^r_{i,t+1}) = \overline{var}_{i,t}(\Delta y_{i,t+1}) + \overline{var}_{i,t}(\pi_{t+1})
# \end{eqnarray}
#
# Not enough information is available for the same kind of transformation of IQR and skewness from nominal to real, so I only use nominal variables. Besides, as there are extreme values on inflation expectations and uncertainty, I also exclude top and bottom $5\%$ of the observations. This further shrinks the sample, when using real moments, to around 40,000. 
#

# #  Perceived income risks: basic facts 
#
#
# ##  Cross-sectional heterogeneity
#
# This section inspects some basic cross-sectional patterns of the subject moments of labor income. In the Figure \ref{fig:histmoms}, I plot the distribution of perceived income risks in nominal and real terms, $\overline{var}_{i,t}$ and $\overline{var^{r}}_{i,t}$, respectively. 
#
# There is a sizable dispersion in perceived income risks. In both nominal and real terms, the distribution is right-skewed with a long tail. Specifically, most of the workers have perceived a variance of nominal earning growth ranging from zero to $20$ (a standard-deviation equivalence of $4-4.5\%$ income growth a year). But in the tail, some of the workers perceive risks to be as high as $7-8\%$ standard deviation a year. To have a better sense of how large the risk is, consider a median individual in our sample, who has an expected earnings growth of $2.4\%$, and a perceived risk of $1\%$ standard deviation. This implies by no means negligible earning risk. \footnote{In the appendix, I also include histograms of expected income growth and subjective skewness, which show intuitive patterns such as nominal rigidity. Besides, about half of the sample exhibits non-zero skewness in their subjective distribution, indicating asymmetric upper/lower tail risks.}  
#
# \begin{center}
# [FIGURE \ref{fig:histmoms} HERE]
# \end{center}
# How are perceived income risks different across a variety of demographic factors? Empirical estimates of income risks of different demographic groups from microdata have been rare\footnote{For instance, <cite data-cite="meghir2004income">(Meghir and Pistaferri (2004))</cite> estimated that high-education group is faced with higher income risks than the low-education group.  <cite data-cite="bloom2018great">(Bloom et al.(2018))</cite> documented that income risks decreases with age and varies with current income level in a U-shaped.}, not mentioning in subjective risk perceptions. Figure \ref{fig:ts_incvar_age} plots the average perceived risks of young, middle-aged, and old workers over the sample period. It is clear that for most of the months, perceived risks decrease with age. Hypothetically, this may be either because of more stable earning dynamics as one is older in the market in reality, or a better grasp of the true income process and higher subjective certainty. The model I will build allows both to play a role.  
#
# \begin{center}
#  [FIGURE \ref{fig:ts_incvar_age} HERE]
#  \end{center}
#
# Another important question is how income risk perceptions depend on the realized income. This is unclear ex-ante because it depends on the true income process as well as the perception formation. SCE does not directly report the current earning by the individual who reports earning forecasts. Instead, I use what's available in the survey, the total pretax household income in the past year as a proxy to the past realizations of labor income. As Figure \ref{fig:barplot_byinc} shows, perceived risks gradually declines as one's household income increases for most range of income. But the pattern reverses for the top income group. Such a non-monotonic relationship between risk perceptions and past realizations, as I will show later in the theoretical section, will be reconciled by people's state-dependent attribution and learning. 
#
# \begin{center}
#  [FIGURE \ref{fig:barplot_byinc} HERE]
# \end{center}
#

# + {"caption": "Distribution of Individual Moments", "code_folding": [], "hide_input": true, "hide_output": true, "label": "fig:histmoms", "note": "this figure plots histograms of the individual income moments. inc for nominal and rinc for real."}
## insert figures

graph_path = os.path.join(path,'../Graphs/ind/')

fig_list = ['hist_incexp.jpg',
            'hist_rincexp.jpg',
            'hist_incvar.jpg',
            'hist_rincvar.jpg']
            
nb_fig = len(fig_list)
    
file_list = [graph_path+ fig for fig in fig_list]

## show figures 
plt.figure(figsize=(18,10))
for i in range(nb_fig):
    plt.subplot(int(nb_fig/2),2,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")

# + {"caption": "Perceived Income by Income", "code_folding": [], "hide_input": true, "hide_output": true, "label": "fig:barplot_byinc", "note": "this figure plots average perceived income risks by the range of household income.", "widefigure": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/ind/')

fig_list = ['boxplot_var_stata.png']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(8,8))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")

# + {"caption": "Perceived Income by Age", "code_folding": [], "hide_input": true, "hide_output": true, "label": "fig:ts_incvar_age", "note": "this figure plots average perceived income risks of different age groups over time.", "widefigure": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/ind/ts/')

fig_list = ['ts_incvar_age_g_mean.png']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(8,8))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")
# -

# ## Counter-cyclicality of perceived risk
#
# Some studies have documented that income risks are counter-cyclical based on cross-sectional income data. \footnote{But they differ in exactly which moments of the income are counter-cyclical. For instance, <cite data-cite="storesletten2004cyclical">Storesletten et al.(2004)</cite> found that variances of income shocks are counter-cyclical, while <cite data-cite="guvenen2014nature"> Guvenen et al.(2014)</cite> and <cite data-cite="catherine_countercyclical_2019">Catherine (2019)</cite>, in contrast, found it to be the left skewness.}  It is worth inspecting if the subjective income risk profile has a similar pattern. Figure \ref{fig:tshe} plots the average perceived income risks from SCE against the YoY growth of the average hourly wage across the United States, which shows a clear negative correlation. Table \ref{macro_corr_he} further confirms such a counter-cyclicality by reporting the regression coefficients of different measures of average risks on the wage rate of different lags. All coefficients are significantly negative. 
#
# \begin{center}
# [FIGURE \ref{fig:tshe} HERE]
# \end{center}
#
#
# \begin{center}
# [TABLE \ref{macro_corr_he} HERE]
# \end{center}
#
# The pattern can be also seen at the state level. Table \ref{macro_corr_he_state} reports the regression coefficients of the monthly average perceived risk within each state on the state labor market conditions, measured by either wage growth or the state-level unemployment rate, respectively. It shows that a tighter labor market (higher wage growth or a lower unemployment rate) is associated with lower perceived income risks. Note that our sample stops in June 2019 thus not covering the outbreak of the pandemic in early 2020. The counter-cyclicality will be very likely more salient if it includes the current period, which was marked by catastrophic labor market deterioration and increase market risks.   
#
# \begin{center}
# [TABLE \ref{macro_corr_he_state} HERE]
# \end{center}
#
# The counter-cyclicality in subjective risk perceptions seen in the survey may suggest the standard assumption of state-independent symmetry in income shocks is questionable. But it may well be, alternatively, because people's subjective reaction to the positive and negative shocks are asymmetric even if the underlying process being symmetric. The model to be constructed in the theoretical section explores the possible role of both. 

# + {"caption": "Recent Labor Market Outcome and Perceived Risks", "code_folding": [], "hide_input": true, "hide_output": true, "label": "fig:tshe", "note": "Recent labor market outcome is measured by hourly earning growth (YoY)."}
## insert figures 
graph_path = os.path.join(path,'../Graphs/pop/')

fig_list = ['tsMeanexp_he.jpg',
            'tsMeanvar_he.jpg']
            
nb_fig = len(fig_list)

file_list = [graph_path+fig for fig in fig_list]


## show figures 

fig, ax = plt.subplots(figsize =(30,10),
                       nrows = nb_fig,
                       ncols = 1)
for i in range(nb_fig):
    ax[i].imshow(mpimg.imread(file_list[i]))
    ax[i].axis("off") 
plt.tight_layout(pad =0, w_pad=0, h_pad=0)

# + {"hide_input": true, "hide_output": true}
macro_corr  = pd.read_excel('../Tables/macro_corr_he.xlsx',index_col=0)
print('Correlation of perceived risks and past labor market conditions')
macro_corr

# + {"hide_input": true, "hide_output": true}
mom_group_state  = pd.read_excel('../Tables/mom_group_state.xls',index_col = 0)
print('Perceived income risk and state labor market condition')
mom_group_state = mom_group_state.replace(np.nan, '', regex=True)
mom_group_state
# -

# ## Experiences and perceived risk 
#
# Different generations also have different perceived income risks. Let us explore to what extent the cohort-specific risk perceptions are influenced by the income volatility experienced by that particular cohort. Different cohorts usually have experienced distinct macroeconomic and individual histories. On one hand, these non-identical experiences could lead to long-lasting differences in realized life-long outcomes. An example is that college graduates graduating during recessions have lower life-long income than others. (<cite data-cite="kahn2010long">Kahn 2010</cite>, <cite data-cite="oreopoulos2012short">Oreopoulos et al. 2012</cite>, <cite data-cite="schwandt2019unlucky">Schwandt and Von Wachter(2019)</cite>). On the other hand, experiences may have also shaped people's expectations directly, leading to behavioral heterogeneity across cohorts (<cite data-cite="malmendier2015learning">Malmendier and Nagel (2015)</cite>). Benefiting from having direct access to the subjective income risk perceptions, I could directly examine the relationship between experiences and perceptions. 
#
# Individuals from each cohort are borned in the same year and obtained the same level of their respective highest education. The experienced volatility specific to a certain cohort $c$ at a given time $t$ can be approximated as the average squared residuals from an income regression based on the historical sample only available to the cohort's life time. This is approximately the unexpected income changes of each person in the sample. I use the labor income panel data from PSID to estimate the income shocks. \footnote{I obtain the labor income records of all household heads between 1970-2017. Farm workers, youth and olds and observations with empty entries of major demographic variables are dropped. } In particular, I first undertake a Mincer-style regression using major demographic variables as regressors, including age, age polynomials, education, gender and time-fixed effect. Then, for each cohort-time sample, the regression mean-squared error (RMSE) is used as the approximate to the cohort/time-specific income volatility. 
#
# There are two issues associated with such an approximation of experienced volatility. First, I, as an economist with PSID data in my hand, am obviously equipped with a much larger sample than the sample size facing an individual that may have entered her experience. Since larger sample also results in a smaller RMSE, my approximation might be smaller than the real experienced volatility. Second, however, the counteracting effect comes from the superior information problem, i.e. the information set held by earners in the sample contains what is not available to econometricians. Therefore, not all known factors predictable by the individual are used as a regressor. This will bias upward the estimated experienced volatility. Despite these concerns, my method serves as a feasible approximation sufficient for my purpose here. 
#
# The right figure in Figure \ref{fig:var_experience_data} plots the (logged) average perceived risk from each cohort $c$ at year $t$ against the (logged) experienced volatility estimated from above. It shows a clear positive correlation between the two, which suggests that cohorts who have experienced higher income volatility also perceived future income to be riskier. The results are reconfirmed in Table \ref{micro_reg}, for which I run a regression of logged perceived risks of each individual in SCE on the logged experienced volatility specific to her cohort while controlling individuals age, income, educations, etc. What is interesting is that the coefficient of $expvol$ declines from 0.73 to 0.41 when controlling the age effect because that variations in experienced volatility are indeed partly from age differences. While controlling more individual factors, the effect of the experienced volatility becomes even stronger. This implies potential heterogeneity as to how experience was translated into perceived risks.     
#
# How does experienced income shock per se affect risk perceptions? We can also explore the question by approximating experienced income growth as the growth in unexplained residuals. As shown in the left figure of Figure \ref{fig:var_experience_data}, it turns out that that a better past labor market outcome experienced by the cohort is associated with lower risk perceptions. This indicates that it is not not just the volatility, but also the change in level of the income, that is assymmetrically extrapolated into their perceiptions of risk. 
#
# \begin{center}
# [FIGURE \ref{fig:var_experience_data} HERE]
# \end{center}
#
# In theory, individual income change is driven by both aggregate and indiosyncratic risks. It is thus worth examining how experienced outcome from the two respective source translate into risk perceptions differently. In order to do so, we need to approximate idiosyncratic and aggregate experiences, separately. The former is basically the unexplained income residual from a regression controlling time fixed effect and also time-education effect. Since the two effects pick up the samplewide or groupwide common factors of each calender year, it excludes aggregate income shocks. The difference between such a residual and one from a regression dropping the two effects can be used to approximate aggregate shocks. As an alternative measure of aggregate economy, I use the official unemployment rate. For all aggregate measures, the volatility is correspondingly computed as the variance across time periods specific to each cohort. 
#
# Figure \ref{fig:experience_id_ag_data} plot income risk perceptions against both aggregate and idiosyncratic  experiences measured by the level and the volatility of shocks. It suggests different patterns between the aggregate and idiosyncratic experiences. In particular, a positive aggregate shock (both indicated by a higher aggregate income growth, or a lower unexmployment rate) is associated with lower risk perceptions. Such a negative relationship seems to be non-existent at the individual level.  What's common between aggregate and idiosyncratic risks is that the volatility of both kinds of experiences are positively correlated with risk perceptions. Such correlations are confirmed in a regression of controlling other individual characteristics, as shown in Table \ref{micro_reg}. Individual volatility, aggregate volatility and experience in unemployment rates are all significantly positively correlated with income risk perceptions. 
#
# \begin{center}
# [FIGURE \ref{fig:experience_id_ag_data} HERE]
# \end{center} 
#
# As another dimension of the inquiry, one may wonder the effects from experiences of income changes of different degree of persistance. In particular, do experiences of volatility from permanent and transitory shocks affect risk perceptions differently? In order to examine this question, what we can do is to decompose the experienced income volatility of different cohorts into components of different degree of persistences and see how the they are loaded into the future perceptions, separately. In particular, I follow the tradition of a long list of labor/macro literature by assuming the unexplained earning to consist of a permanent and a transitory component which have time-varying volatilities \footnote{<cite data-cite="gottschalk1994growth">Gottschalk et al. 1994</cite>, <cite data-cite="carroll1997nature">Carroll and Samwick, 1997</cite>, <cite data-cite="meghir2004income">Meghir and Pistaferri, 2004</cite>, etc.}. Then relying upon the cross-sectional moment restrictions of income changes, one could estimate the size of the permanent and transitory income risks based on realized income data. Experienced permanent and transitory volatility is approximated as the the average of estimated risks of respective component from the year of birth of the cohort till the year for which the income risk perceptions is formed.
#
# In theory, both permanent and transitory risks increase the volatility in income changes in the same way. But the results here suggest the pattern only holds for transitory income risks, as shown in the Figure \ref{fig:experience_var_per_tran_var_data}. In contrast, higher experienced permanent volatility is associated with lower perceived risk. I also confirm that the pattern is not sensitive to the particular definition of the cohort here by alterantively letting people's income risks be specific to education level. To understand which compoennt overweights the other in determining overal risk perception given their opposite signs, I also examine how the relative size of permanent and transitory volatility affect income risk perceptions. The ratio essentially reflect the degree of persistency of income changes, i.e. higher permanent/transitory risk ratio leads to a higher serial correlation of unexpected income changes. The right graph in Figure \ref{fig:experience_var_per_tran_var_data} suggest that higher permanent/transitory ratio is actually associated with lower perceived income risks. The fact that income risk perceptions differ depending upon the nature of experienced volatility suggests that income risk perceptions are formed in a way that is not consistent with the underlying income process. This will be a crutial input to the modeling of perception formation in the next section.
#
# \begin{center}
# [FIGURE \ref{fig:experience_var_per_tran_var_data} HERE]
# \end{center} 
#
#

# + {"caption": "Experienced Volatility and Perceived Income Risk", "code_folding": [], "hide_input": true, "hide_output": true, "label": "fig:var_experience_var_data", "note": "Experienced volatility is the mean squred error(MSE) of income regression based on a particular year-cohort sample. The perceived income risk is the average across all individuals from the cohort in that year."}
## insert figures 
graph_path = os.path.join(path,'../Graphs/ind/')

fig_list = ['experience_var_var_data.png']
            
nb_fig = len(fig_list)

file_list = [graph_path+fig for fig in fig_list]


## show figures 
plt.figure(figsize =(5,5))
for i in range(nb_fig):
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off") 
plt.tight_layout(pad =0, w_pad=0, h_pad=0)

# + {"caption": "Experienced Permanent Volatility and Perceived Income Risk", "code_folding": [], "hide_input": true, "hide_output": true, "label": "fig:experience_var_permanent_var_data", "note": "Experienced permanent volatility is average of the estimated risks of the permanent income component of a particular year-cohort sample. The perceived income risk is the average across all individuals from the cohort in that year."}
## insert figures 
graph_path = os.path.join(path,'../Graphs/ind/')

fig_list = ['experience_var_permanent_var_data.png']
            
nb_fig = len(fig_list)

file_list = [graph_path+fig for fig in fig_list]


## show figures 
plt.figure(figsize =(5,5))
for i in range(nb_fig):
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off") 
plt.tight_layout(pad =0, w_pad=0, h_pad=0)

# + {"caption": "Experienced Transitory Volatility and Perceived Income Risk", "code_folding": [], "hide_input": true, "hide_output": true, "label": "fig:experience_var_transitory_var_data", "note": "Experienced transitory volatility is average of the estimated risks of the transitory component of a particular year-cohort sample. The perceived income risk is the average across all individuals from the cohort in that year."}
## insert figures 
graph_path = os.path.join(path,'../Graphs/ind/')

fig_list = ['experience_var_transitory_var_data.png']
            
nb_fig = len(fig_list)

file_list = [graph_path+fig for fig in fig_list]


## show figures 
plt.figure(figsize =(5,5))
for i in range(nb_fig):
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off") 
plt.tight_layout(pad =0, w_pad=0, h_pad=0)

# + {"caption": "Experienced Permanent/Transitory Ratio and Perceived Income Risk", "code_folding": [], "hide_input": true, "hide_output": true, "label": "fig:experience_var_ratio_var_data", "note": "Experienced permanent/transitory ratio is ratio of the estimated risks of the permanent to transitory component of a particular year-cohort sample. The perceived income risk is the average across all individuals from the cohort in that year."}
## insert figures 
graph_path = os.path.join(path,'../Graphs/ind/')

fig_list = ['experience_var_ratio_var_data.png']
            
nb_fig = len(fig_list)

file_list = [graph_path+fig for fig in fig_list]


## show figures 
plt.figure(figsize =(5,5))
for i in range(nb_fig):
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off") 
plt.tight_layout(pad =0, w_pad=0, h_pad=0)

# + {"hide_input": true, "hide_output": true}
micro_reg_history_vol  = pd.read_excel('../Tables/micro_reg_history_vol.xlsx',index_col = 0)
print('Experienced volatility and perceived income risk')
micro_reg_history_vol = micro_reg_history_vol.replace(np.nan, '', regex=True)
micro_reg_history_vol
# -

#
# ##  Other individual characteristics
#   What other factors are associated with risk perceptions? This section inspects the question by regressing the perceived income risks at the individual level on four major blocks of variables: experiences, demographics, unemployment expectations by the respondent, as well as job-specific characteristics.  The regression is specified as followed. 
#
# \begin{eqnarray}
# \overline{risk}_{i,c,t} = \alpha + \beta_0 \textrm{HH}_{i,c,t} + \beta_1 \textrm{Exp}_{c,t} + \beta_2 \textrm{Prob}_{i,c,t} + \beta_3 \textrm{JobType}_{i,c,t} + \epsilon_{i,t}
# \end{eqnarray}
#
# The dependent variable is the individual $i$ from cohort $c$'s perceived risk. The experience block $\textit{Exp}_{c,t}$ includes individual experienced volatility $\textit{IdExpVol}_{c,t}$, the aggregate experience of volatility $\textit{AgExpVol}_{c,t}$ and experience of unemployment rate $\textit{AgExpUE}_{c,t}$. They are all cohort/time-specific since different birth cohort at different points of time have had difference experience of both micro and macro histories. The second type of factors denoted $\textit{HH}_{i,t}$ represents household-specific demographics such as the age, household income level, education, gender as well as the numeracy of the respondent. In particular, the numeracy score is generated based on the individual's answers to seven questions that are designed to measure the individual's basic knowlege of probability, intrest rate compounding, the difference between nominal and real return and risk diversification. Third, $\textit{Prob}_{i,t}$ represents other subjective probabilities regarding unemployment held by the same individual. As far as this paper is concerned, I include the perceived probability of unemployment herself and the probability of a higher nationwide unemployment rate. The fourth block of factors, as called $\textit{Jobtype}_{i,t}$ includes dummy variables indicating if the job is part-time or if the work is for others or self-employed. 
#
# Besides, since many of the regressors are time-invariant household characteristics, I choose not to control household fixed effects in these regressions ($\omega_i$). Throughout all specifications, I cluster standard errors at the household level because of the concern of unobservable household heterogeneity. 
#
# The regression results reported in Table \ref{micro_reg}  are rather intuitive. From the first to the sixth column, I gradually control more factors. All specifications confirm that higher experienced volatility at both idiosyncratic level and aggregate level, as well as high unemployment rate experience in the past all lead to higher risk perceptions. Besides, workers from low-income households, females, and lower education and self-employed jobs have higher perceived income risks.
#
# In our sample, there are around $15\%$ (6000) of the individuals who report themselves to be self-employed instead of working for others. The effects are statistically and economically significant. Whether a part-time job is associated with higher perceived risk is ambiguous depending on if we control household demographics. At first sight, part-time jobs may be thought of as more unstable. But the exact nature of part-time job varies across different types and populations. It is possible, for instance, that the part-time jobs available to high-income and educated workers bear lower risks than those by the low-income and low-education groups. 
#
# Another interesting finding is that individual risk perception decreases as the individual's numeracy test scores higher. This is not particularly driven by the difference in education as the pattern remains even if we jointly control for the education. This coroborates the findings that individual's perception and decisions are affected by the financial literacy (<cite data-cite="van2011financial">Van Rooij et al. 2011</cite>, <cite data-cite="lusardi2014economic">Lusardi and Mitchell,2014</cite>).
#
# In addition, higher perceived the probability of losing the current job, which I call individual unemployment risk, $\textit{UEprobInd}$ is associated with higher earning risks of the current job. The perceived chance that the nationwide unemployment rate going up next year, which I call aggregate unemployment risk, $\textit{UEprobAgg}$ has a similar correlation with perceived earning risks. Such a positive correlation is important because this implies that a more comprehensively measured income risk facing the individual that incorporates not only the current job's earning risks but also the risk of unemployment is actually higher. Moreover, the perceived risk is higher for those whose perceptions of the earning risk and unemployment risk are more correlated than those less correlated. 
#
#
#  \begin{center}
#  [TABLE \ref{micro_reg} HERE]
#  \end{center}

reg_tb = pd.read_excel('../Tables/micro_reg.xlsx').replace(np.nan,'')

# + {"hide_input": true, "hide_output": true}
reg_tb
# -

# ##  Perceived income risk and decisions
#
# Finally, how individual-specific perceived risks affect household economic decisions such as consumption? The testable prediction is higher perceived risks shall increase precautionary saving motive therefore lower current consumption (higher consumption growth.) Although we cannot directly observe the respondent's spending decisions, we can alternatively rely on the self-reported spending plan in the SCE to shed some light on this \footnote{Other work that directly examines the impacts of expectations on readiness to spend includes <cite data-cite="bachmann2015inflation">Bachmann et al. 2015</cite>,<cite data-cite="coibion2020forward">Coibon et al. 2020</cite>.}.
#
# Table \ref{spending_reg} reports the regression results of planned spending growth over the next year on the expected earning's growth (the first column) as well as a number of perceived income risk measures. \footnote{There is one important econometric concern when I run regressions of the decision variable on perceived risks due to the measurement error in the regressor used here. In a typical OLS regression in which the regressor has i.i.d. measurement errors, the coefficient estimate for the imperfectly measured regressor will have a bias toward zero. For this reason, if I find that willingness to consume is indeed negatively correlated with perceived risks, taking into account the bias, it implies that the correlation of the two is greater in the magnitude.}  Each percentage point increase in expected income growth is associated with a 0.39 percentage point increase in spending growth. At the same time, one percentage point higher in the perceived risk increases the planned spending growth by 0.58 percentage. This effect is even stronger for real income risks. As a double-check, the individual's perceived probability of a higher unemployment rate next year also has a similar effect. These results suggest that individuals do exhibit precautionary saving motives according to their own perceived risks.  
#
#
# \begin{center}
# [TABLE \ref{spending_reg} HERE]
# \end{center}

spending_reg_tb = pd.read_excel('../Tables/spending_reg.xlsx').replace(np.nan,'')

# + {"hide_input": true, "hide_output": true}
spending_reg_tb
# -

# # A life-cycle model with heterogeneous risk perceptions
#
# Each consumer solves a life-cycle consumption/saving problem formulated by <cite data-cite="gourinchas2002consumption">Gourinchas and Parker, 2002</cite>. There is only one deviation from the original model: each agent imperfectly knows the parameters of income process over the life cycle and forms his/her best guess at each point of the time based on past experience. I first set up the model under the assumption of perfect understanding and then extend it to the imperfect understanding scenarior in the next section.   
#
#
#
# ## The standard life-cycle problem
#
# Each agent works for $T$ periods since entering the labor market, during which he/she earns stochastic labor income $y_\tau$ at the work-age of $\tau$. After retiring at age of $T+1$, the agent lives for for another $L-T$ periods of life. Since a cannonical life-cycle problem is the same in nature regardless of the cohort and calender time, we set up the problem generally along the age of work $\tau$. The consumer chooses the whole future consumption path to maximize expected life-long utility. 
#
# \begin{equation}
# \begin{split}
# E\left[\sum^{\tau=L}_{\tau=1}\beta^\tau u(c_{\tau})\right] \\
# u(c) = \frac{c^{1-\rho}}{1-\rho}
# \end{split}
# \end{equation}
#
# where $c_\tau$ represents consumption at the work-age of $\tau$. The felicity function $u$ takes the standard CRRA form with relative risk aversion of $\rho$. We assume away the bequest motive and preference-shifter along life cycle that are present in the original model without loss of the key insights regarding income risks. 
#
# Denote the financial wealth at age of $\tau$ as $b_{\tau}$. Given initial wealth $b_1$, the consumer's problem is subject to the borrowing constraint 
#
# \begin{equation}
# b_{\tau}\geq 0
# \end{equation}
#
# and inter-temporal budget constraint.
#
# \begin{equation}
# \begin{split}
# b_{\tau}+y_{\tau} = m_\tau   \\
# b_{\tau+1} = (m_\tau-c_{\tau})R
# \end{split}
# \end{equation}
#
# where $m_\tau$ is the total cash in hand at the begining of period $\tau$. $R$ is the risk-free interest rate. Note that after retirement labor income is zero through the end of life. 
#
# The stochastic labor income during the agent's career consists of a mulplicative predictable component by known factors $Z_\tau$ and a stochastic component $\epsilon_{\tau}$ which embodies shocks of different nature.  
#
# \begin{equation}
# \begin{split}
# y_{\tau} = \phi Z_{\tau}\epsilon_{\tau} 
# \end{split}
# \end{equation}
#
#
# Notice here I explicitly include the predictable component, deviating from the common practice in the literature. Although under perfect understanding, the predictable component does not enter consumption decision effectively since it is anticipated ex ante, this is no longer so once we introduce imperfect understanding regarding the parameters of the income process $\phi$. The prediction uncertainty enters the perception of income risks. We will return to this point in the next section. 
#
#
# The stochastic shock to income $\epsilon$ is composed of two components: a permanent one $p_t$  and a transitory one $u$. The former grows by a age-specific growth rate $G$ along the life cycle and is subject to a shock $n$ at each period. 
#
# \begin{equation}
# \begin{split}
# \epsilon_{\tau} = p_{\tau}u_{\tau} \\
# p_{\tau} = G_{\tau}p_{\tau-1} n_{\tau}
# \end{split}
# \end{equation}
#
# The permanent shock $n$ follows a log normal distribution, $ln(n_\tau) \sim N(0,\sigma^2_\tau)$. The transitory shock $u$ either takes value of zero with probability of $0\leq p<1$, i.e. unemployment, or otherwise follows a log normal with $ln(u_\tau) \sim N(0,\sigma^2_u)$. Following <cite data-cite="gourinchas2002consumption">Gourinchas and Parker, 2002</cite>, I assume the size of the volatility of the two shocks are time-invariant. The results of this paper are not sensitive to this assumption. 
#
# At this stage, we do not seek to differentiate the aggregate/idiosyncratic components of either one the two enter the individual consumption problem indifferently under perfect understanding. With imperfect understanding and subjective attribution, however, the differences of the two matters since it affects the prediction uncertainty and perceived income risks. 
#
# The following value function characterizes the problem. 
#
# \begin{equation}
# \begin{split}
# V_{\tau}(m_\tau, p_\tau) = \textrm{max} \quad u(c_\tau) + \beta E_{\tau}\left[V_{\tau+1}(m_{\tau+1}, p_{\tau+1})\right] 
# \end{split}
# \end{equation}
#
# where the agents treat total cash in hand and permanent income as the two state variables. On the backgroud, the income process parameters $\Gamma = [\phi,\sigma_n,\sigma_u]$ affect the consumption decisions. But to the extent that the agents have perfect knowledge of them, they are simply taken as given. 
#
#
# ## Under heterogenous perceptions
#
# The crucial deviation of this model from the standard framework reproduced above is that the agents do not know about the income parameters $\Gamma$, and the decisions are only based on their best guess obtained through learning from experience in a manner we formulate in the previous section. This changes the problem in at least two ways. First, given agents potentially differ in their experiences, perceived income processes differ. Second, even if under same experiences, different subjective determinations of the nature of income shocks result in different risk perceptions. To allow for the cross-sectional heterogeneity across individuals and cohorts in income risk perceptions, now explicitly define the problem using agent-time-cohort-specific value function. For agent $i$ from cohort $c=t-\tau$ at time $t$, the value function is the following.
#
# \begin{equation}
# \begin{split}
# V_{i,\tau,t}(m_{i,\tau,t}, p_{i,\tau,t}) = \textrm{max} \quad u(c_{i,\tau+1,t+1}) + \beta E_{i,\tau,t}\left[V_{i,\tau+1,t+1}(m_{i,\tau+1,t+1}, p_{i,\tau+1,t+1})\right] 
# \end{split}
# \end{equation}
#
# Notice that the key difference of the new value function from the one under a perfect understanding is that expectational operator of next-period value function becomes subjective and potentially agent-time-specific. Another way to put it is that $E_{i,\tau,t}$ is conditional on the most recent parameter estimate of the income process $\tilde \Gamma_{i,\tau,t} = \left[\tilde \phi_{i,\tau,t},\tilde \sigma_{n,i,\tau,t}, \tilde \sigma_{u,i,\tau,t}\right]$ and the uncertainty about the estimate $Var_{i,\tau,t}(\tilde \phi)$. 
#
# The perceived income risk affects the value of the expected value. It implicitly contains two components. The first is the shock uncertainty that can be predicted at best by the past income volatility estimation of different components. The second is the parameter uncertainty regarding the agents' estimation of a parameter associated with the deterministic components $\phi$. Since both components imply a further dispersion from the perfect understanding case, it will unambiguously induce a stronger precautionary saving motive than the latter case. 
#
#
# ## Consumption functions 
#
# I compare the life cycle consumption functions between
#
# - perfectly understanding vs imperfect understanding
# - same age different experiences 
# - under different digree of attribution 
#
# ## Implications for consumption inequality 
#
# - consumption inequality(thus wealth inequality) and heterogeneity in MPCs now is further amplified by belief differences in income risks. 

#
# #  Conclusion
#
# How do people form perceptions about their income risks? Theoretically, this paper builds an experience-based learning model that features an imperfect understanding of the size of the risks as well as its nature. By extending the learning-from-experience into a cross-sectional setting, I introduce a mechanism in which future risk perceptions are dependent upon past income volatility or cross-sectional distributions of the income shocks. I also introduce a novel channel - subjective attribution, into the learning to capture how income risk perceptions are also affected by the subjective determination of the nature of income risks. It is shown that the model generates a few testable predictions about the relationship between experience/age/income and perceived income risks.
#
# Empirically, I utilize a recently available panel of income density surveys of U.S. earners to shed light directly on subjective income risk profiles. I explore the cross-sectional heterogeneity in income risk perceptions across ages, generations, and income group as well as its cyclicality with the current labor market outcome. I found that risk perceptions are positively correlated with experienced income volatility, therefore differing across age and cohorts. I also found perceived income risks of earners counter-cyclically react to the recent labor market conditions.
#
# Finally, the paper builds the experience-based-learning and subjective attribution into an otherwise standard life cycle model of consumption. I show an imperfect understanding of the income process unambiguously motivates additional precautionary saving than in a model of perfect understanding. I also show that the consumption decisions of agents at the same age may still differ as long as they have experienced different histories at both individual and aggregate levels. Such belief heterogeneity further amplifies the inequality in consumption and wealth accumulation of different generations.
#
# Many interesting questions are worth exploring although they are beyond the scope of the paper. First, to what extent the model in this paper could help account for the well-documented differences between millennials and earlier generations in their saving behaviors, homeownership, and stock market investment? Within the very short span of their early career, millennials have had experienced two aggregate economic catastrophes, namely the global financial crisis and the pandemic. The evidence and model in this paper both suggest that this may have persistent impacts on the new generations' risk perceptions, thus economic decisions. 
#
# Second, what general equilibrium consequences would the life-cycle and intergenerational differences in risk perceptions generate? It is true that demographic compositions are slow-moving variables. But the gradual change in the demographic structure of the economy may interact with the different experienced macroeconomic histories, generating non-stationary belief distributions across time. This will undoubtedly lead to a different macroeconomic equilibrium. 
#
# Third, although this paper focuses on the size and nature of income risks, the imperfect understanding could also and may very likely take the form of misperceiving correlation between different random variables relevant to economic decisions. A perfect example of this is the correlation between income risks and stock market returns. The subjective correlation between the two may shed light on participation puzzles and equity premium in the macroeconomic finance literature. 
