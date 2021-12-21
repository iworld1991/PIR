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
## this python script read over all latex tables from the folder and insert it in a latex file
# -

import os


# + {"code_folding": [0]}
def insertbefore(oldtex, key, tex_to_add):
    """ 
    Inserts 'tex_to_add' into 'oldtex' right before 'key'. 
    """
    i = oldtex.find(key)
    return oldtex[:i] + '   '  + tex_to_add +'       ' + oldtex[i:]


# +
## look over files first

cwd = os.getcwd()
ltxtb_where = os.path.join(cwd,'../Tables/latex/')
ltx_name = 'PerceivedIncomeRisk.tex'
ltx_file = os.path.join(cwd,ltx_name)

# +
## get all latex codes for talbes 
ltxtbs = ''   # strings 

for file in os.listdir(ltxtb_where):
    if file.endswith(".tex"):
        file_path = os.path.join(ltxtb_where, file)
        f = open(file_path, 'r')
        ltxtb = f.read()
        #print(len(ltxtb))
        ltxtbs = ltxtbs + ltxtb

# +
## write it in the master latex file 

key = '% Add a bibliography block'

with open(ltx_file,'r') as f:
    old_tex = str(f.read())
    #print(old_tex)
    f.close()
    
new_tex = insertbefore(old_tex,
                       key,
                       ltxtbs)
#print(new_tex)
# -

with open(ltx_file,'w') as f:
    f.write(new_tex)
    f.close()


