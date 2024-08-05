import pickle
import streamlit as st
from rdkit import Chem
from streamlit_ketcher import st_ketcher
from molfeat.calc import FPCalculator
from lightgbm import LGBMRegressor
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

The ”IrLumDB App” is an ML-based service to predict phosphorescence wavelength of bis-cyclometalated iridium(III) complexes requiring only molecular formula of the ligands as a feature. Please enter SMILES of the ligands (or draw the structural formula in the corresponding window) and press **“Predict maximum wavelength (nm)”** button to perform the prediction.

Usage notes:
* The desired complexes usually contain two cyclometalated ligands and one ancillary ligand; thus L1 and L2 should correspond to the cyclometalated ligands and L3 should correspond to the ancillary ligand.

* Some ligands make formally covalent bonds with the Ir(III) ion. For these a negatively charged bond-forming atom should be drawn in the SMILES of corresponding ligand.
---
### To get SMILES of your ligand, draw custom molecule and click **"Apply"** button.
''')
smile_code = st_ketcher('[c-]1ccccc1-c1ccccn1', height=400)
st.markdown(f"""### Your SMILES: ``{smile_code}``
Copy and paste this SMILES into the corresponding box below:""")

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

model = pickle.load(open('lgbm.pkl', 'rb'))

if st.button("Predict maximum wavelength(nm)"):
    if L1 and L2 and L3:
        # try:
        L_res = calc(Chem.MolFromSmiles(L1)) + calc(Chem.MolFromSmiles(L2)) + calc(Chem.MolFromSmiles(L3))
        L_res = L_res.reshape(1, -1)
        col1.image(draw_molecule(L1), caption=L1)
        col2.image(draw_molecule(L2), caption=L2)
        col3.image(draw_molecule(L3), caption=L3)
        pred = str(round(model.predict(L_res), 1))
        st.markdown(f'**{pred} nm**')
        # except:
        #     st.error("Incorrect SMILES entered")

    else:
        st.error("Please enter all three ligands")
