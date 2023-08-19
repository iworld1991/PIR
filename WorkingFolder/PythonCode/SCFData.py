# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## [Survey of Consumer Finance (SCF)](https://www.federalreserve.gov/econres/scfindex.htm) data cleaning and analysis 
#
# - the code builds on this [blog](https://notebook.community/DaveBackus/Data_Bootcamp/Code/Lab/SCF_data_experiment_Brian)

import numpy as np
import pandas as pd   #The data package
import sys            #The code below wont work for any versions before Python 3. This just ensures that (allegedly).


# +
## figure plotting configurations

import matplotlib.pyplot as plt 

plt.style.use('seaborn-v0_8')
plt.rcParams["font.family"] = "Times New Roman" #'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['axes.labelweight'] = 'bold'

## Set the 
plt.rc('font', size=15)
# Set the axes title font size
plt.rc('axes', titlesize=15)
# Set the axes labels font size
plt.rc('axes', labelsize=15)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=15)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=15)
# Set the legend font size
plt.rc('legend', fontsize=15)
# Set the font size of the figure title
plt.rc('figure', titlesize=15)
# -

import requests
import io
import zipfile      #Three packages we'll need to unzip the data
import matplotlib.pyplot as plt                            


# + code_folding=[0]
def unzip_survey_file(year = '2013'):
    """
    The next two lines of code converts the URL into a format that works
    with the "zipfile" package.
    """
    if int(year) <1989:
        url = 'http://www.federalreserve.gov/econres/files/'+year+'_scf'+year[2:]+'bs.zip'
        #old url = 'http://www.federalreserve.gov/econresdata/scf/files/'+year+'_scf'+year[2:]+'bs.zip'

    else: 
        url = 'http://www.federalreserve.gov/econres/files/scfp'+year+'s.zip'    
    url = requests.get(url)
    """
    Next, zipfile downloads, unzips, and saves the file to your computer. 'url2013_unzipped' 
    contains the file path for the file.
    """
    url_unzipped = zipfile.ZipFile(io.BytesIO(url.content))
    return url_unzipped.extract(url_unzipped.namelist()[0])


# + code_folding=[]
## keep the code for each year for future use

## which year? choose from 1983,1992,2001,2013,2016,2019 
year = '2016'  

df2016 = pd.read_stata(unzip_survey_file(year=year))

## to rename some variables for 1983 vintage 
#df1983 = df1983.rename(columns = {'b3201':'income', 'b3324':'networth', 'b3015' : 'wgt'})
#df1983 = df1983[df1983['income']>=0]
# -

list(df2016.columns)

# ## Variable definitions 
#
# - [code book for 2016](https://sda.berkeley.edu/sdaweb/docs/scfcomb2016/DOC/hcbk0007.htm)
# - The [definition](https://www.federalreserve.gov/econres/files/Networth%20Flowchart.pdf) of networth 

# + code_folding=[]
## make new variables 
#df2016['lqwealth'] = df2016['liq']+df2016['govtbnd']+df2016['obnd']+df2016['stocks']+df2016['nmmf'] - df2016['ccbal'] 
## Kaplan, Violante, and Weidner (2014)/Econmetrica paper definition

df2016['lqwealth'] = df2016['liq']+df2016['govtbnd'] - df2016['ccbal'] 

### filters and clean variables 
df2016 = df2016[(df2016['age']>=25) & (df2016['age']<=85)]
df2016 = df2016[df2016['wageinc']>0]

df2016 = df2016[df2016['income']>0]
df2016 = df2016[df2016['norminc']>0]
## drop negative liquid wealth 
df2016 = df2016[df2016['lqwealth']>=0]

## compute log values 
df2016['lwage_income'] = np.log(df2016['wageinc'])
df2016['lincome'] = np.log(df2016['income'])
df2016['lnorminc'] = np.log(df2016['norminc'])

## compute ratios 
df2016['w2wage_income']= df2016['networth']/ df2016['wageinc']
df2016['lw2wage_income']= df2016['lqwealth']/ df2016['wageinc']

df2016['w2income']= df2016['networth']/ df2016['norminc']
df2016['lw2income']= df2016['lqwealth']/ df2016['norminc']


# + code_folding=[0]
## age polynomials regressions 

df2016['age2'] = df2016['age']**2 
df2016['age3'] = df2016['age']**3
df2016['age4'] = df2016['age']**4

import statsmodels.api as sm 
import statsmodels.formula.api as smf

model = smf.ols(formula = 'lnorminc~ age+age2+age3+age4',
                data = df2016)
results = model.fit()
df2016['lnorminc_pr'] = results.predict()

# -
# ### Cross-sectional distribution of income and wealth

# +
## distribution in monetary values 
import seaborn as sns

data_plot = df2016[['norminc','wgt']][df2016['norminc']<df2016['norminc'].quantile(0.95)]

dist = sns.displot(data = data_plot,
            x = 'norminc',
            weights = 'wgt',
            kde=True,
            stat = 'density',
            bins = 100).set(title='Distribution of annual permanent income (2016 SCF)',
                            xlabel='Usual annual income (USD)')

norminc_av = (data_plot['norminc']*data_plot['wgt']).sum()/data_plot['wgt'].sum()
print('mean permanent income: $', str(round(norminc_av,3)))

dist.fig.set_size_inches(8,6)

# + code_folding=[4]
## distribution in monetary values 

data_plot = df2016[['wageinc','wgt']][df2016['wageinc']<df2016['wageinc'].quantile(0.95)]

dist = sns.displot(data = data_plot,
            x = 'wageinc',
            weights = 'wgt',
            kde=True,
            stat = 'density',
            bins = 100).set(title='Distribution of wage income (2016 SCF)',
                            xlabel='Wage income (USD)')

wageinc_av = (data_plot['wageinc']*data_plot['wgt']).sum()/data_plot['wgt'].sum()
print('mean wage income: $', str(round(wageinc_av,3)))

dist.fig.set_size_inches(8,6)

# +
data_plot = df2016[['lqwealth','wgt']][df2016['lqwealth']<df2016['lqwealth'].quantile(0.95)]

dist = sns.displot(data = data_plot,
            x = 'lqwealth',
            weights = 'wgt',
            kde=True,
            stat = 'density',
            color = 'brown',
            bins = 100).set(title='Distribution of net liquid wealth (2016 SCF)',
                            xlabel='Liquid wealth (USD)'
)

lqwealth_av = (data_plot['lqwealth']*data_plot['wgt']).sum()/data_plot['wgt'].sum()

print('mean net liquid wealth: $', str(lqwealth_av))

dist.fig.set_size_inches(8,6)

# +
data_plot = df2016[['lw2income','wgt']][df2016['lw2income']<df2016['lw2income'].quantile(0.95)]

dist = sns.displot(data = data_plot,
            x = 'lw2income',
            weights = 'wgt',
            kde=True,
            stat = 'density',
            color = 'blue',
            bins = 100).set(title='Distribution of net-liquid-wealth/permanent-income ratio (2016 SCF)',
                            xlabel='Liquid wealth/permanent income ratio')

lw2income_av = (data_plot['lw2income']*data_plot['wgt']).sum()/data_plot['wgt'].sum()
print('mean net liquid wealth/income ratio: ', str(round(lw2income_av,2)))

dist.fig.set_size_inches(8,6)
# -

# ### (Liquid) Wealth Inequality 

# +
from Utility import cal_ss_2markov,lorenz_curve, gini

SCF_lqwealth, SCF_lqweights = np.array(df2016['lqwealth']), np.array(df2016['wgt'])

## get the lorenz curve weights of liquid wealth from SCF 
SCF_lqwealth_sort_id = SCF_lqwealth.argsort()
SCF_lqwealth_sort = SCF_lqwealth[SCF_lqwealth_sort_id]
SCF_lqweights_sort = SCF_lqweights[SCF_lqwealth_sort_id]
SCF_lqweights_sort_norm = SCF_lqweights_sort/SCF_lqweights_sort.sum()

SCF_lq_share_agents_ap, SCF_lq_share_ap = lorenz_curve(SCF_lqwealth_sort,
                                                 SCF_lqweights_sort_norm,
                                                 nb_share_grid = 200)

## gini 

gini_lq_SCF = gini(SCF_lq_share_agents_ap,
                 SCF_lq_share_ap)


# +
## plot lorenz curve 
fig, ax = plt.subplots(figsize=(6,6))

ax.plot(SCF_lq_share_agents_ap,
        SCF_lq_share_ap, 'r-.',
        label='SCF,Gini={}'.format(round(gini_lq_SCF,2)))
ax.plot(SCF_lq_share_ap,
        SCF_lq_share_ap, 
        'k-',
        label='equality curve')

ax.legend()
plt.xlim([0,1])
plt.ylim([0,1])


# + code_folding=[2, 57]
## two other functions that give Lorenz curve more details 

def weighted_percentiles(data, variable, weights, percentiles = [], 
                         dollar_amt = False, subgroup = None, limits = []):
    """
    data               specifies what dataframe we're working with
    
    variable           specifies the variable name (e.g. income, networth, etc.) in the dataframe
    
    percentiles = []   indicates what percentile(s) to return (e.g. 90th percentile = .90)
    
    weights            corresponds to the weighting variable in the dataframe
    
    dollar_amt = False returns the percentage of total income earned by that percentile 
                       group (i.e. bottom 80% of earners earned XX% of total)
                         
    dollar_amt = True  returns the $ amount earned by that percentile (i.e. 90th percentile
                       earned $X)
                         
    subgroup = ''      isolates the analysis to a particular subgroup in the dataset. For example
                       subgroup = 'age' would return the income distribution of the age group 
                       determined by the limits argument
                       
    limits = []        Corresponds to the subgroup argument. For example, if you were interesting in 
                       looking at the distribution of income across heads of household aged 18-24,
                       then you would input "subgroup = 'age', limits = [18,24]"
                         
    """
    import numpy 
    a  = list()
    data[variable+weights] = data[variable]*data[weights]
    if subgroup is None:
        tt = data
    else:
        tt = data[data[subgroup].astype(int).isin(range(limits[0],limits[1]+1))] 
    values, sample_weight = tt[variable], tt[weights]
    
    for index in percentiles: 
        values = numpy.array(values)
        index = numpy.array(index)
        sample_weight = numpy.array(sample_weight)

        sorter = numpy.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        weighted_percentiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_percentiles /= numpy.sum(sample_weight)
        a.append(numpy.interp(index, weighted_percentiles, values))
    
    if dollar_amt is False:    
        return[tt.loc[tt[variable]<=a[x],
                      variable+weights].sum()/tt[variable+weights].sum() for x in range(len(percentiles))]
    else:
        return a
    
    
def figureprefs(data, 
                variable = 'income', 
                labels = False, 
                legendlabels = []):
    
    percentiles = [i * 0.05 for i in range(20)]+[0.99, 1.00]

    fig, ax = plt.subplots(figsize=(6,6));

    ax.set_xticks([i*0.1 for i in range(11)]);       #Sets the tick marks
    ax.set_yticks([i*0.1 for i in range(11)]);

    vals = ax.get_yticks()                           #Labels the tick marks
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals]);
    ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals]);

    ax.set_title('Lorenz Curve: United States in 2016',  #Axes titles
                  fontsize=18, loc='center');
    ax.set_ylabel('Cumulative Percent', fontsize = 12);
    ax.set_xlabel('Percent of Agents', fontsize = 12);
    
    if type(data) == list:
        values = [weighted_percentiles(data[x], variable,
                    'wgt', dollar_amt = False, percentiles = percentiles) for x in range(len(data))]
        for index in range(len(data)):
            plt.plot(percentiles,values[index],
                     linewidth=2.0, marker = 's',clip_on=False,label=legendlabels[index]);
            for num in [10, 19, 20]:
                ax.annotate('{:3.1f}%'.format(values[index][num]*100), 
                    xy=(percentiles[num], values[index][num]),
                    ha = 'right', va = 'center', fontsize = 12);

    else:
        values = weighted_percentiles(data, variable,
                    'wgt', dollar_amt = False, percentiles = percentiles)
        plt.plot(percentiles,values,
                     linewidth=2.0, marker = 's',clip_on=False,label=legendlabels);

    plt.plot(percentiles,percentiles, linestyle =  '--', color='k',
            label='Perfect Equality');
   
    plt.legend(loc = 2)


# + code_folding=[]
## plot a different version of lorenz curve with percentile of wealth 

years_graph = [df2016]
labels = ['2016']

figureprefs(years_graph, 
            variable = 'networth', 
            legendlabels = labels)
figureprefs(years_graph, 
            variable = 'lqwealth', 
            legendlabels = labels);
# -

# ### Life-cycle wealth and income profile 

import joypy
from matplotlib import cm

# + code_folding=[0, 1]
#labels=[y if y%10==0 else None for y in list(df2016.age.unique())]
fig, axes = joypy.joyplot(df2016, 
                          by="age", 
                          column= "lwage_income", 
                          #labels=labels, 
                          range_style='own', 
                          grid="y", 
                          linewidth=1, 
                          legend=False, 
                          figsize=(6,20),
                          title="Wage income over life cycle",
                          colormap=cm.summer)

# + code_folding=[]
#labels=[y if y%10==0 else None for y in list(df2016.age.unique())]
fig, axes = joypy.joyplot(df2016, 
                          by="age", 
                          column= "lnorminc", 
                          #labels=labels, 
                          range_style='own', 
                          grid="y", 
                          linewidth=1, 
                          legend=False, 
                          figsize=(6,20),
                          title="Permanent income over life cycle",
                          colormap=cm.summer)

# +
## Life cycle income / wealth profiles 

age = df2016['age'].unique()

wm = lambda x: np.average(x, weights=df2016.loc[x.index, "wgt"])

age_av_wealth = df2016.groupby('age').agg(av_wealth = ('networth',wm))
age_med_wealth = df2016.groupby('age').agg(med_wealth=('networth','median'))

age_av_lqwealth = df2016.groupby('age').agg(av_lqwealth = ('lqwealth',wm))
age_med_lqwealth = df2016.groupby('age').agg(med_lqwealth=('lqwealth','median'))

age_av_w2i = df2016.groupby('age').agg(av_w2i = ('w2income',wm))
age_med_w2i = df2016.groupby('age').agg(med_w2i=('w2income','median'))


age_av_lqw2i = df2016.groupby('age').agg(av_lqw2i = ('lw2income',wm))
age_med_lqw2i = df2016.groupby('age').agg(med_lqw2i=('lw2income','median'))

age_av_lincome = df2016.groupby('age').agg(av_lincome = ('lincome',wm))
age_med_lincome = df2016.groupby('age').agg(med_lincome=('lincome','median'))

age_av_lnorminc = df2016.groupby('age').agg(av_lnorminc = ('lnorminc',wm))
age_med_lnorminc = df2016.groupby('age').agg(med_lnorminc=('lnorminc','median'))

age_av_lwage_income = df2016.groupby('age').agg(av_lwage_income= ('lwage_income',wm))
age_med_lwage_income = df2016.groupby('age').agg(med_lwage_income= ('lwage_income','median'))



# +
plt.title('Net wealth over life cycle')
plt.plot(np.log(age_av_wealth),label='average net wealth')
plt.plot(np.log(age_med_wealth),label='median net wealth')
plt.plot(np.log(age_av_lqwealth),label='average net liquid wealth')
plt.plot(np.log(age_med_lqwealth),label='median net liquid wealth')

plt.legend(loc=0)
# -

plt.title('Net wealth over life cycle')
plt.plot(np.log(age_av_w2i),label='average net wealth/income ratio')
plt.plot(np.log(age_med_w2i),label='median net wealth/income ratio')
plt.plot(np.log(age_av_lqw2i),label='average net liquid wealth/income ratio')
plt.plot(np.log(age_med_lqw2i),label='median net liquid wealth/income ratio')
plt.legend(loc=0)

# +
plt.title('Income over life cycle')

plt.plot(np.log(age_av_lwage_income),label='wage income')
plt.plot(np.log(age_av_lincome),label='average income')
plt.plot(np.log(age_med_lincome),label='median income')
plt.legend(loc=0)
# -

plt.title('Permanent income over life cycle')
plt.plot(np.log(age_av_lnorminc),label='average income')
plt.plot(np.log(age_med_lnorminc),label='median income')
plt.legend(loc=0)

# + code_folding=[]
## merge all age profiles 

to_merge = [age_med_wealth,
            age_av_lqwealth,
            age_med_lqwealth,
            age_av_w2i,
            age_med_w2i,
            age_av_lqw2i,
            age_med_lqw2i,
            age_av_lincome,
            age_med_lincome,
            age_av_lnorminc,
            age_med_lnorminc,
           age_av_lwage_income,
           age_med_lwage_income]

SCF_age_profile = age_av_wealth

for  df in to_merge:
    SCF_age_profile = pd.merge(SCF_age_profile,
                              df,
                              left_index=True,
                              right_index=True)
    
SCF_age_profile.to_pickle('data/SCF_age_profile.pkl')
# -

SCF_age_profile


