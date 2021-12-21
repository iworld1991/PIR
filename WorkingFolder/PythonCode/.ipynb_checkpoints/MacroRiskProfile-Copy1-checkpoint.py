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

# ## Perceived Labor Income Risks and Asset Returns
#
#
# - This notebook first downloads asset return indicators 
# - Then we examine the correlation of higher moments of labor income risks and asset returns 
# - It also inspects the cross-sectional pattern of the subjective moments

# +
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import statsmodels.api as sm 
import scipy.stats as st  
from statsmodels.graphics import tsaplots as tsplt

# +
plt.style.use('ggplot')
from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('pdf','png','jpg')
#plt.rcParams['savefig.dpi'] = 75

#plt.rcParams['figure.autolayout'] = False
#plt.rcParams['figure.figsize'] = 10, 6
#plt.rcParams['axes.labelsize'] = 18
#plt.rcParams['axes.titlesize'] = 20
#plt.rcParams['font.size'] = 16
#plt.rcParams['lines.linewidth'] = 2.0
#plt.rcParams['lines.markersize'] = 8
#plt.rcParams['legend.fontsize'] = 14

#plt.rcParams['text.usetex'] = True
#plt.rcParams['font.family'] = "serif"
#plt.rcParams['font.serif'] = "cm"
# -

pd.options.display.float_format = '{:,.2f}'.format

# ###  1. Download stock return series 

# + {"code_folding": []}
## s&p 500 series 

start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2020, 3, 30)

# + {"code_folding": []}
## downloading the data from Fred
sp500D= web.DataReader('sp500', 'fred', start, end)
vixD = web.DataReader('VIXCLS','fred',start,end)
# -

vixD.plot(lw = 2)
vixplt = plt.title('vix')

# + {"code_folding": []}
#plotting
#sp500D.plot(lw=2)
#sp500plt = plt.title('S&P 500')

# +
## collapse to monthly data 
sp500D.index = pd.to_datetime(sp500D.index)
sp500M = sp500D.resample('M').last()

vixD.index = pd.to_datetime(vixD.index)
vixM = vixD.resample('M').mean()
# -

sp500M.plot(lw = 3)
#sp500Mplt = plt.title('S&P 500 (end of month)')

sp500MR = np.log(sp500M).diff(periods = 12)

sp500MR.plot(lw = 3 )
sp500MRplt = plt.title('Monthly return of S&P 500')

# ###  2. Loading and cleaning perceived income series 

# + {"code_folding": []}
## loading the stata file
SCEProbIndM = pd.read_stata('../SurveyData/SCE/IncExpSCEProbIndM.dta')
SCEDstIndM = pd.read_stata('../SurveyData/SCE/IncExpSCEDstIndM.dta')

# + {"code_folding": []}
## subselect the dataframe
sub_var = ['date',
           'userid',
           'Q24_var',
           'Q24_mean',
           'Q24_iqr',
           'Q24_rmean',
           'Q24_rvar']
IncSCEIndMoms = SCEProbIndM[sub_var]


sub_var2 = ['date','userid','IncVar','IncMean','IncSkew','IncKurt']
IncSCEIndMomsEst = SCEDstIndM[sub_var2]

## drop nan observations
IncSCEIndMoms = IncSCEIndMoms.dropna(how='any')
IncSCEIndMomsEst = IncSCEIndMomsEst.dropna(how='any')

# +
## rename 

IncSCEIndMoms = IncSCEIndMoms.rename(columns={'Q24_mean': 'incexp',
                                               'Q24_var': 'incvar',
                                               'Q24_iqr': 'inciqr',
                                               'Q24_rmean':'rincexp',
                                               'Q24_rvar': 'rincvar'})


# + {"code_folding": []}
## deal with clusterring skewness value around zero first

#IncSCEIndMomsEst['IncSkew'] = IncSCEIndMomsEst['IncSkew'].copy().replace(0,np.nan)

# + {"code_folding": []}
moms = ['incexp','incvar','inciqr','rincexp','rincvar']
moms_est = ['IncSkew','IncKurt']

## compute population summary stats for these ind moms
IncSCEPopMomsMed = pd.pivot_table(data = IncSCEIndMoms, 
                                  index=['date'], 
                                  values = moms,
                                  aggfunc= 'median').reset_index().rename(columns={'incexp': 'meanMed', 
                                                                                   'incvar': 'varMed',
                                                                                   'inciqr': 'iqrMed',
                                                                                   'rincexp':'rmeanMed',
                                                                                   'rincvar':'rvarMed'})

IncSCEPopMomsMean = pd.pivot_table(data = IncSCEIndMoms, 
                                   index = ['date'],
                                   values = moms,
                                   aggfunc = 'mean').reset_index().rename(columns={'incexp': 'meanMean',
                                                                                 'incvar': 'varMean',
                                                                                 'inciqr': 'iqrMean',
                                                                                 'rincexp':'rmeanMean',
                                                                                 'rincvar':'rvarMean'})

IncSCEPopMomsEstMed = pd.pivot_table(data = IncSCEIndMomsEst, 
                                     index = ['date'],
                                     values = moms_est,
                                     aggfunc = 'median').reset_index().rename(columns={'IncMean': 'meanEstMed',
                                                                                      'IncVar': 'varEstMed',
                                                                                      'IncSkew': 'skewEstMed',
                                                                                      'IncKurt':'kurtEstMed'})

IncSCEPopMomsEstMean = pd.pivot_table(data = IncSCEIndMomsEst, 
                                      index=['date'],
                                      values = moms_est,
                                      aggfunc= 'mean').reset_index().rename(columns={'IncMean': 'meanEstMean',
                                                                                    'IncVar': 'varEstMean',
                                                                                    'IncSkew': 'skewEstMean',
                                                                                    'IncKurt':'kurtEstMean'})
# -

T = len(IncSCEPopMomsEstMean)

# ### 3. Cross-sectional patterns of subjective distributions
#

# + {"code_folding": []}
### histograms 

for mom in moms:
    fig,ax = plt.subplots(figsize=(6,4))
    sns.distplot(IncSCEIndMoms[mom],
                 kde = True,
                 color ='red',
                 bins = 20) 
    plt.xlabel(mom, fontsize = 12)
    plt.xlabel(mom,fontsize = 12)
    plt.ylabel("Frequency",
               fontsize = 12)
    plt.savefig('../Graphs/ind/hist_'+str(mom)+'.jpg')

# +
## for estimated moments 

for mom in moms_est:
    fig,ax = plt.subplots(figsize=(6,4))
    mom_nonan = IncSCEIndMomsEst[mom].dropna()
    mom_lb, mom_ub = np.percentile(mom_nonan,2),np.percentile(mom_nonan,98) ## exclude top and bottom 3% observations
    #print(mom_lb)
    #print(mom_ub)
    to_keep = (mom_nonan < mom_ub) & (mom_nonan > mom_lb) & (mom_nonan!=0)
    #print(to_keep.shape)
    mom_nonan_truc = mom_nonan[to_keep]
    #print(mom_nonan_truc.shape)
    sns.distplot(mom_nonan_truc,
                 kde = True, 
                 color = 'red',
                 bins = 18) 
    plt.xlabel(mom, fontsize = 13)
    plt.ylabel("Frequency",fontsize = 13)
    plt.savefig('../Graphs/ind/hist'+str(mom)+'.jpg')
    
# -

# ### 4. Combinine the two series 

# + {"code_folding": []}
## streamline the dates for merging 

# adjusting the end-of-month dates to the begining-of-month for combining 
sp500MR.index = sp500MR.index.shift(1,freq='D') 
vixM.index = vixM.index.shift(1,freq='D')


IncSCEPopMomsMed.index = pd.DatetimeIndex(IncSCEPopMomsMed['date'] ,freq='infer')
IncSCEPopMomsMean.index = pd.DatetimeIndex(IncSCEPopMomsMean['date'] ,freq='infer')
IncSCEPopMomsEstMed.index = pd.DatetimeIndex(IncSCEPopMomsEstMed['date'] ,freq='infer')
IncSCEPopMomsEstMean.index = pd.DatetimeIndex(IncSCEPopMomsEstMean['date'] ,freq='infer')

# + {"code_folding": []}
dt_combM = pd.concat([sp500MR,
                      vixM,
                      IncSCEPopMomsMed,
                      IncSCEPopMomsMean,
                      IncSCEPopMomsEstMed,
                      IncSCEPopMomsEstMean],
                     join="inner",
                     axis=1).drop(columns=['date'])
# -

dt_combM.tail()

# +
## save sp500 as stata for other analysis 

sp500MR.to_stata('../OtherData/sp500.dta')

# + {"code_folding": []}
## date index for panel 

IncSCEIndMoms.index = IncSCEIndMoms['date']
IncSCEIndMoms.index.name = None
IncSCEIndMomsEst.index = IncSCEIndMomsEst['date']
IncSCEIndMomsEst.index.name = None

# + {"code_folding": []}
## merge individual moments and macro series 

dt_combIndMs = pd.merge(IncSCEIndMoms,
                        IncSCEIndMomsEst,
                        on = ['date','userid'])
dt_combIndMs.index = dt_combIndMs['date']

dt_combMacroM = pd.merge(sp500MR,
                         vixM,
                         left_index = True,
                         right_index = True)

dt_combIndM = pd.merge(dt_combMacroM,
                       dt_combIndMs,
                       left_index = True,
                       right_index = True)
# -

# ### 5. Seasonal adjustment (not successful yet)

# +
#to_sa_test = ['meanMed']
#to_sa_list = list(dt_combM.columns.drop('sp500'))

# + {"code_folding": []}
## inspect for seasonal pattern by looking into the autocovariance
"""
for sr in to_sa_list:
    tsplt.plot_acf(dt_combM[sr],
                   lags = np.arange(T-20),
                   alpha = 0.03)
    plt.title('Autocorrelation of '+ sr)
    plt.savefig('../Graphs/pop/acf_'+str(sr)+'.jpg')
    
"""
# -

# - Judging from the acf plots, it seems that only the population mean and median of expected earning growth has seasonal patterns at 12 month frequency, higher moments such as variance, skewness does not have this pattern.   

# + {"code_folding": [0, 2]}
"""
for sr in to_sa_test:
    series = dt_combM[sr]
    samodel = sm.tsa.UnobservedComponents(series,
                                          level='fixed intercept',
                                          seasonal = 12)
    res = samodel.fit(disp = True)
    print(res.summary())
    #res_plt = res.plot_components(figsize=(4,11))
    #plt.plot(res.level.filtered)
    
"""
# -

# ### 6. Correlation with stock market returns and times series patterns 

corr_table = dt_combM.corr()
corr_table.to_excel('../Tables/corrM.xlsx')
corr_table

# + {"code_folding": [4, 18]}
## try different lags or leads

lead_loop = 12

def pval_str(pval):
    if pval < 0.01:
        star = '***'
    elif pval >= 0.01 and pval<0.05:
        star = '**'
    elif pval >= 0.05 and pval <= 0.1:
        star = '*'
    else:
        star = ''
    return star

def corrtostr(corr): 
    return str(round(corr[0],2)) + str(pval_str(corr[1]))

def corrprint(corr,
              var):
    print('correlation coefficient betwen sp500 and median'+
              str(var) + 
              ' is ' + 
              str(round(corr[0],2)) +
              ', and p-value is ' + 
              str(round(corr[1],2))
             )

corr_list = []
col_list = []

#print('median')
for moms in ['var','iqr','rvar']:
    col_list.append('median:'+str(moms))
    for lead in range(lead_loop):
        corr = st.pearsonr(np.array(dt_combM['sp500'][lead+1:]),
                           np.array(dt_combM[str(moms)+'Med'])[:-(lead+1)]
                          )
        corr_str = corrtostr(corr)
        corr_list.append(corr_str)
        #corrprint(corr, moms)

for moms in ['skew']:
    col_list.append('median:'+str(moms))
    for lead in range(lead_loop):
        corr = st.pearsonr(np.array(dt_combM['sp500'][lead+1:]),
                           np.array(dt_combM[str(moms)+'EstMed'])[:-(lead+1)]
                          )
        corr_str = corrtostr(corr)
        corr_list.append(corr_str)
        #corrprint(corr,moms)
        
#print('mean')

for moms in ['var','iqr','rvar']:
    col_list.append('mean:'+str(moms))
    for lead in range(lead_loop):
        corr = st.pearsonr(np.array(dt_combM['sp500'][lead+1:]),
                           np.array(dt_combM[str(moms)+'Mean'])[:-(lead+1)]
                          )
        corr_str = corrtostr(corr)
        corr_list.append(corr_str)
        #corrprint(corr,moms)
        
for moms in ['skew']:
    col_list.append('mean:'+str(moms))
    for lead in range(lead_loop):
        corr = st.pearsonr(np.array(dt_combM['sp500'][lead+1:]),
                           np.array(dt_combM[str(moms)+'EstMean'])[:-(lead+1)]
                          )
        corr_str = corrtostr(corr)
        corr_list.append(corr_str)
        #corrprint(corr,moms)
        
             
corr_array = np.array(corr_list).reshape([int(len(corr_list)/lead_loop),
                                          lead_loop])
corr_df = pd.DataFrame(corr_array,
                       index = col_list)
# -

corr_df.T

# + {"code_folding": []}
## output tables to latex 

beginningtex = """
\\begin{table}[ht]
\\centering
\\begin{adjustbox}{width={\\textwidth}}
\\begin{threeparttable}
\\caption{Correlation between Perceived Income Risks and Stock Market Return}
\\label{macro_corr}
"""

endtex = """\\begin{tablenotes}
\item *** p$<$0.001, ** p$<$0.01 and * p$<$0.05. 
\item This table reports correlation coefficients between different perceived income moments(inc for nominal 
and rinc for real) at time 
$t$ and the monthly s\&p500 return by the end of $t+k$ for $k=0,1,..,11$. 
\\end{tablenotes}
\\end{threeparttable}
\\end{adjustbox}
\\end{table}"""

## write to latex 

f = open('../Tables/latex/macro_corr.tex', 'w')
f.write(beginningtex)
tb_ltx = corr_df.T.to_latex().replace('llllll','cccccc')
f.write(tb_ltx)
f.write(endtex)
f.close()
# -

## output table to excel 
corr_df.to_excel('../Tables/macro_corr.xlsx')

# + {"code_folding": []}
## plots of correlation for MEDIAN population stats 

figsize = (80,40)
lw = 20
fontsize = 70

for i,moms in enumerate( ['mean','var','iqr','rmean','rvar']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM.index, dt_combM['sp500'], color='gray', width=25,label='SP500')
    ax2.plot(dt_combM[str(moms)+'Med'], 
             'r--',
             lw = lw,
             label=str(moms)+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc = 0,
              fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax2.legend(loc=2,
               fontsize = fontsize)
    ax2.set_ylabel(moms,fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsMed'+str(moms)+'.jpg')
    #cor,pval =st.pearsonr(np.array(dt_combM['sp500']),
    #                      np.array(dt_combM[str(moms)+'Med']))
    #print('Correlation coefficient is '+str(round(cor,3)) + ', p-value is '+ str(round(pval,3)))
    

for i,moms in enumerate( ['skew']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM.index, 
           dt_combM['sp500'], 
           color='gray', 
           width=25,
           label='SP500')
    ax2.plot(dt_combM[str(moms)+'EstMed'], 
             'r--',
             lw = lw,
             label=str(moms)+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc=0,
              fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax2.legend(loc = 2,
               fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsEstMed'+str(moms)+'.jpg')
    #cor,pval = st.pearsonr(np.array(dt_combM['sp500']),
    #                      np.array(dt_combM[str(moms)+'EstMed']))
    #print('Correlation coefficient is '+str(round(cor,3)) + ', p-value is '+ str(round(pval,3)))

# + {"code_folding": [2, 27]}
## plots of correlation for Mean population stats 

for i,moms in enumerate( ['mean','var','iqr','rmean','rvar']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM.index, 
           dt_combM['sp500'], 
           color='gray', 
           width=25,
           label='SP500')
    ax2.plot(dt_combM[str(moms)+'Mean'], 
             'r--',
             lw = lw,
             label=str(moms)+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc=0,
             fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax2.legend(loc=2,
              fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsMean'+str(moms)+'.jpg')
    
    #cor,pval = st.pearsonr(np.array(dt_combM['sp500']),
    #                      np.array(dt_combM[str(moms)+'Mean']))
    #print('Correlation coefficient is '+str(round(cor,3)) + ', p-value is '+ str(round(pval,3)))
    
for i,moms in enumerate( ['skew']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM.index, 
           dt_combM['sp500'], 
           color='gray', 
           width=25,
           label='SP500')
    ax2.plot(dt_combM[str(moms)+'EstMean'], 
             'r--',
             lw = lw,
             label=str(moms)+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc=0,
              fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax2.legend(loc=2,
              fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsEstMean'+str(moms)+'.jpg')
    #cor,pval =st.pearsonr(np.array(dt_combM['sp500']),
    #                      np.array(dt_combM[str(moms)+'EstMean']))
    #print('Correlation coefficient is '+str(round(cor,3) ) + ', p-value is '+ str(round(pval,3) ))

# + {"code_folding": []}
## moving average 

dt_combM3mv = dt_combM.rolling(3).mean()
# -

crr3mv_table = dt_combM3mv.corr()
crr3mv_table.to_excel('../Tables/corr3mvM.xlsx')
crr3mv_table

# + {"code_folding": [0]}
## plots of correlation for 3-month moving MEDIAN average 

for i,moms in enumerate( ['mean','var','iqr','rmean','rvar']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM3mv.index, 
           dt_combM3mv['sp500'], 
           color='gray', 
           width=25,
           label='SP500')
    ax2.plot(dt_combM3mv[str(moms)+'Med'], 
             'r--',
             lw = lw,
             label=str(moms)+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc=0,
              fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax2.legend(loc=2,
              fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsMed3mv'+str(moms)+'.jpg')
    #cor,pval = st.pearsonr(np.array(dt_combM3mv['sp500']),
    #                      np.array(dt_combM3mv[str(moms)+'Med']))
    #print('Correlation coefficient is '+str(cor) + ', p-value is '+ str(pval))

for i,moms in enumerate( ['skew']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM3mv.index, 
           dt_combM3mv['sp500'], 
           color = 'gray', 
           width = 25,
           label = 'SP500')
    ax2.plot(dt_combM3mv[str(moms)+'EstMed'], 
             'r--',
             lw = lw,
             label=str(moms)+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc=0,
              fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax2.legend(loc=2,
              fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsEstMed3mv'+str(moms)+'.jpg')
    #cor,pval =st.pearsonr(np.array(dt_combM3mv['sp500']),
    #                      np.array(dt_combM3mv[str(moms)+'EstMed']))
    #print('Correlation coefficient is '+str(round(cor,3) ) + ', p-value is '+ str(round(pval,3) ))

# + {"code_folding": [2]}
## plots of correlation for 3-month moving mean average 

for i,moms in enumerate( ['mean','var','iqr','rmean','rvar']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM3mv.index, 
           dt_combM3mv['sp500'], 
           color='gray', 
           width=25,
           label='SP500')
    ax2.plot(dt_combM3mv[str(moms)+'Mean'], 
             'r--',
             lw = lw,
             label=str(moms)+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc=0,
             fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax2.legend(loc=2,
             fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsMean3mv'+str(moms)+'.jpg')
    #cor,pval =st.pearsonr(np.array(dt_combM['sp500']),
    #                      np.array(dt_combM3mv[str(moms)+'Mean']))
    #print('Correlation coefficient is '+str(cor) + ', p-value is '+ str(pval))
    
    
for i,moms in enumerate( ['skew','kurt']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM3mv.index, 
           dt_combM3mv['sp500'], 
           color='gray', 
           width=25,
           label='SP500')
    ax2.plot(dt_combM3mv[str(moms)+'EstMean'], 
             'r--',
             lw = lw, 
             label=str(moms)+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc=0,
             fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax2.legend(loc=2,
             fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsEstMean3mv'+str(moms)+'.jpg')
    #cor,pval =st.pearsonr(np.array(dt_combM['sp500']),
    #                      np.array(dt_combM3mv[str(moms)+'EstMean']))
    #print('Correlation coefficient is '+str(cor) + ', p-value is '+ str(pval))
# -

# ### 7. Individual regressions 

# + {"code_folding": []}
lead = 2

for i,moms in enumerate( ['incexp','incvar','inciqr','rincvar','IncSkew']):
    print(moms)
    Y = np.array(dt_combIndM[moms])[lead+1:]
    X = np.array(dt_combIndM['sp500'])[:-(lead+1)]
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    rs = model.fit()
    print(rs.summary())
    
# -






