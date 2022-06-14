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

# +
#############################################################################
## This python script executes all the codes that generate all the         ##
##  up-to-date results seen in the paper draft.                            ##
##  Specifically, it includes following tasks                              ##
##  1. Importing, cleaning SCE data and undertakes the density estimtation ##
##  2. Using the estimated moments as inputs, analyze cross-sectional      ##
#        patterns of these moments.                                        ##
#   3. Examine the correlation between moments and stock market returns.   ##
#   4. Incorporates all the results in the master notebook.                ##
#   5. Convert the notebook into a latex and then compile it as a          ##
#   publishable pdf.                                                       ##
#############################################################################
#   When executed, the code first prompt a line  asking for user's         ##
#      input on if density estimation is needed?                           ##
#       'yes':  takes about 25-minutes - 30 minutes                        ##
#       or 'no': takes the stored estimates as input.                      ##
#############################################################################

import os
import sys

# Find pathname to this file:
my_file_path = os.path.dirname(os.path.abspath("do_all.py"))

# -

## python codes to run
DoDensityEst_py =  os.path.join(my_file_path,"DoDensityEst.py")
MicroRiskProfile_py = os.path.join(my_file_path,"MicroRiskProfile.py")
MacroRiskProfile_py = os.path.join(my_file_path,"MacroRiskProfile.py")
PerceivedIncomeRisk_py = os.path.join(my_file_path,"PerceivedIncomeRisk.py")
TheoryLearning_py = os.path.join(my_file_path,"TheoryLearning.py")


do_denst = str(input("Do you want to redo density estimation?"
                     "\n if yes, it takes about 30mins; if no, existing estimates are used, taking only 2-3 mins."))

# + {"code_folding": []}

if do_denst =='yes':
    print("Started running the codes.")
    print('1. Do density estimation of SCE.')
    exec(open(DoDensityEst_py).read())
    print('2. Individual level analysis')
    exec(open(MicroRiskProfile_py).read())
    print('3. Macro analysis')
    exec(open(MacroRiskProfile_py).read())
    print('4. Theory and Simulation')
    exec(open(TheoryLearning_py).read())
    print('5. Compile master notebook')
    exec(open(PerceivedIncomeRisk_py).read())
    print('6. Compile the latex file.')

elif do_denst == 'no':
    print("Started running the codes.")
    print('2. Individual level analysis')
    exec(open(MicroRiskProfile_py).read())
    print('3. Macro analysis')
    exec(open(MacroRiskProfile_py).read())
    print('4. Theory and Simulation')
    exec(open(TheoryLearning_py).read())
    print('5. Compile master notebook')
    exec(open(PerceivedIncomeRisk_py).read())

else:
    print('Wrong input. Need to type yes or no.')

print('Complete.\n Now, ready to run bash script "to_tex.sh" to compile the most recent PerceivedIncomeRisk.pdf' )
