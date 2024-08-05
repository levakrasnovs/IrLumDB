import pandas as pd
import pickle
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher
from molfeat.calc import FPCalculator
from xgboost import XGBRegressor

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.Draw.MolToImage(mol)

def canonize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

calc = FPCalculator("ecfp")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.set_page_config(layout="wide")

# Web App Title
st.markdown('''
# **IrLumDB App**

The ”IrLumDB App” is an ML-based service to predict luminescence wavelength of bis-cyclometalated iridium(III) complexes requiring only molecular formula of the ligands as a feature. Please enter SMILES of the ligands (or draw the structural formula in the corresponding window) and press **“Predict maximum wavelength (nm)”** button to perform the prediction.

Usage notes:
* The desired complexes usually contain two cyclometalated ligands and one ancillary ligand; thus L1 and L2 should correspond to the cyclometalated ligands and L3 should correspond to the ancillary ligand.

* Some ligands make formally covalent bonds with the Ir(III) ion. For these a negatively charged bond-forming atom should be drawn in the SMILES of corresponding ligand.

* The ML model uses only spectroscopic data obtained in dichloromethane solvent, thus the predicted luminescence wavelength is aimed to be also in **dichloromethane solution** of the corresponding complex.
---
### To get SMILES of your ligand, draw custom molecule and click **"Apply"** button or copy SMILES from popular ligands:
''')

exp = st.expander("Popular ligands")
exp1col, exp2col, exp3col = exp.columns(3)
with exp:
    exp1col.markdown('### ppy(-)')
    exp1col.image(draw_molecule('[c-]1ccccc1-c1ccccn1'), caption="""[c-]1ccccc1-c1ccccn1""")
    exp2col.markdown('### dfppy(-)')
    exp2col.image(draw_molecule('Fc1c[c-]c(-c2ccccn2)c(F)c1'), caption='Fc1c[c-]c(-c2ccccn2)c(F)c1')
    exp3col.markdown('### piq(-)')
    exp3col.image(draw_molecule('[c-]1ccccc1-c1nccc2ccccc12'), caption='[c-]1ccccc1-c1nccc2ccccc12')
    exp1col.markdown('### bzq(-)')
    exp1col.image(draw_molecule('[c-]1cccc2ccc3cccnc3c12'), caption='[c-]1cccc2ccc3cccnc3c12')
    exp2col.markdown('### bpy')
    exp2col.image(draw_molecule('c1ccc(-c2ccccn2)nc1'), caption='c1ccc(-c2ccccn2)nc1')
    exp3col.markdown('### phen')
    exp3col.image(draw_molecule('c1cnc2c(c1)ccc1cccnc12'), caption='c1cnc2c(c1)ccc1cccnc12')
    exp1col.markdown('### pq(-)')
    exp1col.image(draw_molecule('[c-]1ccccc1-c1ccc2ccccc2n1'), caption='[c-]1ccccc1-c1ccc2ccccc2n1')
    exp2col.markdown('### bphen')
    exp2col.image(draw_molecule('c1ccc(-c2ccnc3c2ccc2c(-c4ccccc4)ccnc23)cc1'), caption='c1ccc(-c2ccnc3c2ccc2c(-c4ccccc4)ccnc23)cc1')
    exp3col.markdown('### dppz')
    exp3col.image(draw_molecule('c1ccc2nc3c4cccnc4c4ncccc4c3nc2c1'), caption='c1ccc2nc3c4cccnc4c4ncccc4c3nc2c1')
    exp1col.markdown('### acac(-)')
    exp1col.image(draw_molecule('CC(=O)/C=C(/C)[O-]'), caption='CC(=O)/C=C(/C)[O-]')
    exp2col.markdown('### picolinate')
    exp2col.image(draw_molecule('O=C([O-])c1ccccn1'), caption='O=C([O-])c1ccccn1')
    exp3col.markdown('### tmd(-)')
    exp3col.image(draw_molecule('O=C(/C=C([O-])/C(C)(C)C)C(C)(C)C'), caption='O=C(/C=C([O-])/C(C)(C)C)C(C)(C)C')
    exp1col.markdown('### dmbpy')
    exp1col.image(draw_molecule('Cc1ccnc(-c2cc(C)ccn2)c1'), caption='Cc1ccnc(-c2cc(C)ccn2)c1')

smile_code = st_ketcher('[c-]1ccccc1-c1ccccn1', height=500)
st.markdown(f"""### Your SMILES:""")
st.code(smile_code, language="")
st.markdown(f"""### Copy and paste this SMILES into the corresponding box below:""")

form = st.form(key="form_settings")
col1, col2, col3 = st.columns(3)

L1 = col1.text_input(
        "SMILES L1",
        placeholder='Cc1cc2nc(-c3[c-]cccc3)n(Cc3ccccc3)c2cc1C',
        key='L1')

L2 = col2.text_input(
        "SMILES L2",
        placeholder='Cc1cc2nc(-c3[c-]cccc3)n(Cc3ccccc3)c2cc1C',
        key='L2')

L3 = col3.text_input(
        "SMILES L3",
        placeholder='CC(=O)/C=C(/C)[O-]',
        key='L3')

model = XGBRegressor()
model.load_model('xgboost_model.json')

df = pd.read_csv('BigIrDB_v15.csv')
df['L1'] = df['L1'].apply(lambda x: canonize_smiles(x))
df['L2'] = df['L2'].apply(lambda x: canonize_smiles(x))
df['L3'] = df['L3'].apply(lambda x: canonize_smiles(x))

if st.button("Predict maximum wavelength(nm)"):
    if L1 and L2 and L3:
        mol1 = Chem.MolFromSmiles(L1)
        mol2 = Chem.MolFromSmiles(L2)
        mol3 = Chem.MolFromSmiles(L3)
        if (mol1 is not None) & (mol2 is not None) & (mol3 is not None):
            canonize_l1 = Chem.MolToSmiles(mol1)
            canonize_l2 = Chem.MolToSmiles(mol2)
            canonize_l3 = Chem.MolToSmiles(mol3)
            search_df = df[(df['L1'] == canonize_l1) & (df['L2'] == canonize_l2) & (df['L3'] == canonize_l3)]
            if search_df.shape[0] == 0:
                L_res = calc(mol1) + calc(mol2) + calc(mol3)
                L_res = L_res.reshape(1, -1)
                col1.image(draw_molecule(L1), caption=L1)
                col2.image(draw_molecule(L2), caption=L2)
                col3.image(draw_molecule(L3), caption=L3)
                pred = str(round(model.predict(L_res)[0], 1))
                st.markdown(f'# {pred} nm')
            else:
                st.markdown(f'### Found this complex in IrLumDB:')
                col1search, col2search, col3search, col4search = st.columns(4)
                col1search.markdown(f'**λlum,nm**')
                col2search.markdown(f'**Solvent:**')
                col3search.markdown(f'**Abbreviation in the source:**')
                col4search.markdown(f'**Source**')
                for lam, solvent, doi, abbr in zip(search_df['λlum,nm'], search_df['solvent'], search_df['DOI'], search_df['Abbreviation_in_the_article']):
                    col1search.markdown(f'**{lam} nm**')
                    col2search.markdown(f'**{solvent}**')
                    col3search.markdown(f'**{abbr}**')
                    col4search.markdown(f'**https://doi.org/{doi}**')

        else:
            st.error("Incorrect SMILES entered")

    else:
        st.error("Please enter all three ligands")
