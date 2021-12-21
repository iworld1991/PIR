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

# +
## this python script read over all latex tables from the folder and insert it in a latex file
# -

import os


# + {"code_folding": []}
## a function that is used to insert latex codes to the latex file

def insertbefore(oldtex, key, tex_to_add):
    """
    Inserts 'tex_to_add' into 'oldtex' right before 'key'.
    """
    i = oldtex.find(key)
    return oldtex[:i] + '   '  + tex_to_add +'       ' + oldtex[i:]


def replaceafter(oldtex, key, tex_to_add):
    """
    Inserts 'tex_to_add' into 'oldtex' right after 'key'.
    """
    i = oldtex.find(key)
    howlong = len(key)
    return '   ' + oldtex[:i+howlong] + '   '  + tex_to_add

# +
## look over files first

cwd = os.getcwd()
ltxtb_where = os.path.join(cwd,'../Tables/latex/')
tb_ltx_name = 'latex/table_figures.tex'
tb_ltx_file = os.path.join(cwd,tb_ltx_name)
ltx_name = 'latex/PerceivedIncomeRisk.tex'
ltx_file = os.path.join(cwd,ltx_name)

# +
## get all latex codes for tables
ltxtbs = ''   # strings

for file in os.listdir(ltxtb_where):
    if file.endswith(".tex"):
        file_path = os.path.join(ltxtb_where, file)
        f = open(file_path, 'r')
        ltxtb = f.read()
        #print(len(ltxtb))
        ltxtbs = ltxtbs +'\n'+'\clearpage'+'\n'+ltxtb

key1 = '% tables'

with open(tb_ltx_file,'r') as f:
    old_tb_tex = str(f.read())
    #print(old_tex)
    f.close()


new_tex = replaceafter(old_tb_tex,
                      key1,
                      ltxtbs)
# -

with open(tb_ltx_file,'w') as f:
    f.write(new_tex)
    f.close()

# +
## write it in the master latex file

key2 = '% Add a bibliography block'

with open(ltx_file,'r') as f:
    old_tex = str(f.read())
    #print(old_tex)
    f.close()

tb_lines = '\input{./table_figures.tex}'

new_tex = insertbefore(old_tex,
                       key2,
                       tb_lines)
# -

with open(ltx_file,'w') as f:
    f.write(new_tex)
    f.close()
