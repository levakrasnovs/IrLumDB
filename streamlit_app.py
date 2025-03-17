import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from xgboost import XGBRegressor
from streamlit_ketcher import st_ketcher
from molfeat.calc import FPCalculator

def hamming_distance(fp1, fp2):
    return np.sum(fp1 != fp2)

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.Draw.MolToImage(mol)

def canonize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def check_ligands(mol1, mol2, mol3):
    allowed_atoms = ["C", "O", "N", "H", "Cl", "F", "S", "P"]
    def contains_only_allowed_atoms(mol):
        return any(atom.GetSymbol() not in allowed_atoms for atom in mol.GetAtoms())

    canonize_l1 = Chem.MolToSmiles(mol1)
    canonize_l2 = Chem.MolToSmiles(mol2)
    canonize_l3 = Chem.MolToSmiles(mol3)

    if (len(mol1.GetAtoms()) < 6) | (len(mol2.GetAtoms()) < 6) | (len(mol3.GetAtoms()) < 6):
        st.error("Only ligands with more than 5 atoms are available for input.")
        return False
    elif (contains_only_allowed_atoms(mol1) | contains_only_allowed_atoms(mol2) | contains_only_allowed_atoms(mol3)):
        st.error("The model can predict molecules containing atoms: C, O, N, Cl, F, S, P.")
        return False
    elif ('[c-]' not in canonize_l1) | ('[c-]' not in canonize_l2):
        st.error("The complex should contain TWO cyclometalated ligands, i.e. TWO ligands with deprotonated carbon as L1 and L2.")
        return False
    elif canonize_l1 == canonize_l2 == canonize_l3:
        st.error("The complex should contain TWO cyclometalated ligands, i.e. TWO ligands with deprotonated carbon as L1 and L2. Your query contains deprotonated carbon in the L3 section. Please correct it.")
        return False
    else:
        return True

calc = FPCalculator("ecfp")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.set_page_config(page_title='IrLumDB', layout="wide")

df = pd.read_csv('IrLumDB.csv')
lum = df[['Max_wavelength(nm)', 'PLQY', 'Solvent', 'DOI', 'PLQY_in_train']]
lum = lum[lum['PLQY_in_train'] != 0]
lum = lum[~lum['PLQY'].isna()]
lum = lum[lum['Solvent'].apply(lambda x: x in ['CH2Cl2', 'CH3CN', 'toluene', 'CH3OH', 'THF'])]

df['L1_mol'] = df['L1'].apply(Chem.MolFromSmiles)
df['L2_mol'] = df['L2'].apply(Chem.MolFromSmiles)
df['L3_mol'] = df['L3'].apply(Chem.MolFromSmiles)
df[f'L1_ecfp'] = df['L1_mol'].apply(lambda x: calc(x))
df[f'L2_ecfp'] = df['L2_mol'].apply(lambda x: calc(x))
df[f'L3_ecfp'] = df['L3_mol'].apply(lambda x: calc(x))

df_pred = pd.read_csv('benz_online.csv')

col1intro, col2intro = st.columns([2, 1])
col1intro.markdown("""
# IrLumDB App v1.0

The ”IrLumDB App” is an ML-based service integrated with the experimental database to predict luminescence wavelength (**λlum**) and photoluminescence quantum yield (**PLQY**) of bis-cyclometalated iridium(III) complexes requiring only molecular formula of the ligands as a feature.

### There are currently two operation modes:
* exploration of the database (**“explore”** window)
* prediction of **λlum** and **PLQY** (**“search and predict”** window)

If you are using our database or the App please cite: **Towards Accelerating the Discovery of Efficient Iridium(III) Emitters Using Novel Database and Machine Learning Based Only on Structural Formula**, J. Mater. Chem. C, 2025, https://doi.org/10.1039/D5TC00305A.

Download dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13987455.svg)](https://doi.org/10.5281/zenodo.13987455)
""")

col2intro.image('TOC.png')

tabs = st.tabs(["Explore", "Search and Predict", "Predicted complexes (Work in progress...)"])

with tabs[0]:
    fig_lum = px.scatter(lum, x="Max_wavelength(nm)", y="PLQY", color="Solvent", hover_data={'DOI': True}, title='Space of photophysical properties for bis-cyclometalated iridium(III) complexes')
    fig_lum.update_layout(yaxis_title='PLQY')
    st.plotly_chart(fig_lum)

    fig_qy = px.histogram(lum, x='PLQY', nbins=64, title='PLQY distribution in the IrLumDB')
    fig_qy.update_layout(yaxis_title='Number of entries')
    fig_qy.update_layout(xaxis_title='PLQY')
    st.plotly_chart(fig_qy)

    fig = px.histogram(df, x='Max_wavelength(nm)', nbins=64, title='Maximum wavelength(nm) distribution in the IrLumDB')
    fig.update_layout(yaxis_title='Number of entries')
    st.plotly_chart(fig)

    st.markdown('The “IrLumDB” database contains data about **1454** experimentally measured luminescence spectra of **1287** unique iridium(III) complexes reported in the **340** literature papers. To explore the database, please choose the desired emission wavelength interval below:')
    min_value = 435
    max_value = 938
    initial_value = (500, 600)

    slider_value = st.slider(
        label="λlum,nm",
        min_value=min_value,
        max_value=max_value,
        value=initial_value
    )

    sort_param = st.radio(
        "Sort data by:",
        ["λlum,nm", "PLQY"])

    if st.button("Set wavelength range"):
        if sort_param == "λlum,nm":
            range_df = df[(df['Max_wavelength(nm)'] <= slider_value[1]) & (df['Max_wavelength(nm)'] >= slider_value[0])].sort_values(by='Max_wavelength(nm)')
        else:
            range_df = df[(df['Max_wavelength(nm)'] <= slider_value[1]) & (df['Max_wavelength(nm)'] >= slider_value[0])].sort_values(by='PLQY', ascending=False)
        num = str(range_df.shape[0])
        st.success(f"Selected range: {slider_value}. Found {num} entries:")
        col1range, col2range, col3range, col4range, col5range, col6range, col7range, col8range = st.columns([1, 1, 1, 2, 2, 2, 2, 2])
        col1range.markdown(f'**λlum,nm**')
        col2range.markdown(f'**PLQY**')
        col3range.markdown(f'**Solvent:**')
        col4range.markdown(f'**Abbreviation in the source:**')
        col5range.markdown(f'**Source**')
        col6range.markdown(f'**L1**')
        col7range.markdown(f'**L2**')
        col8range.markdown(f'**L3**')

        for lam, plqy, solvent, doi, abbr, L1, L2, L3 in zip(range_df['Max_wavelength(nm)'],
                                                           range_df['PLQY'],
                                                           range_df['Solvent'],
                                                           range_df['DOI'],
                                                           range_df['Abbreviation_in_the_article'],
                                                           range_df['L1'],
                                                           range_df['L2'],
                                                           range_df['L3']):

            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 2, 2, 2, 2, 2])
            col1.markdown(f'**{lam} nm**')
            col2.markdown(f'**{plqy}**')
            col3.markdown(f'**{solvent}**')
            col4.markdown(f'**{abbr}**')
            col5.markdown(f'**https://doi.org/{doi}**')
            col6.image(draw_molecule(L1), caption=L1)
            col7.image(draw_molecule(L2), caption=L2)
            col8.image(draw_molecule(L3), caption=L3)

with tabs[1]:

    st.markdown("""Please enter SMILES of the ligands (or draw the structural formula in the corresponding window) and press “**Search in the database and predict properties**” button to perform the prediction. If the complex exists in the database, experimental data will be displayed. If the complex does not exist in the database, the predicted **λlum** and **PLQY** will appear.

Usage notes:
* The desired complexes usually contain two cyclometalated ligands and one ancillary ligand; thus L1 and L2 should correspond to the cyclometalated ligands and L3 should correspond to the ancillary ligand.
* Some ligands make formally covalent bonds with the Ir(III) ion. For these a negatively charged bond-forming atom should be drawn in the SMILES of corresponding ligand.
* The ML model uses only spectroscopic data obtained in **dichloromethane solvent**, thus the predicted **λlum** and **PLQY** is aimed to be also in dichloromethane solution of the corresponding complex.

    ### To get SMILES of your ligand, draw custom molecule and click **"Apply"** button or copy SMILES from popular ligands:""")

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
        exp2col.markdown('### diMeNHC')
        exp2col.image(draw_molecule('Cn1[c-][n+](C[n+]2[c-]n(C)cc2)cc1'), caption='Cn1[c-][n+](C[n+]2[c-]n(C)cc2)cc1')

    smile_code = st_ketcher(height=400)
    st.markdown(f"""### Your SMILES:""")
    st.markdown(f"``{smile_code}``")
    st.markdown(f"""### Copy and paste this SMILES into the corresponding box below:""")

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

    model_lum = XGBRegressor()
    model_lum.load_model('xgboost_lum.json')
    model_plqy = XGBRegressor()
    model_plqy.load_model('xgboost_plqy.json')

    if st.button("Search in the database and predict properties"):
        if L1 and L2 and L3:
            mol1 = Chem.MolFromSmiles(L1.strip())
            mol2 = Chem.MolFromSmiles(L2.strip())
            mol3 = Chem.MolFromSmiles(L3.strip())
            if (mol1 is not None) & (mol2 is not None) & (mol3 is not None):
                if check_ligands(mol1, mol2, mol3):
                    canonize_l1 = Chem.MolToSmiles(mol1)
                    canonize_l2 = Chem.MolToSmiles(mol2)
                    canonize_l3 = Chem.MolToSmiles(mol3)
                    col1.image(draw_molecule(L1), caption=L1)
                    col2.image(draw_molecule(L2), caption=L2)
                    col3.image(draw_molecule(L3), caption=L3)
                    search_df = df[(df['L1'] == canonize_l1) & (df['L2'] == canonize_l2) & (df['L3'] == canonize_l3)]
                    if search_df.shape[0] == 0:
                        L1_res_ecfp = calc(mol1)
                        L2_res_ecfp = calc(mol2)
                        L3_res_ecfp = calc(mol3)
                        L_res = L1_res_ecfp + L2_res_ecfp + L3_res_ecfp
                        L_res = L_res.reshape(1, -1)
                        pred_lum = str(int(round(model_lum.predict(L_res)[0], 0)))
                        pred_plqy = round(model_plqy.predict(L_res)[0]*100, 1)
                        str_plqy = str(pred_plqy)
                        predcol1, predcol2 = st.columns(2)
                        predcol1.markdown(f'## Predicted luminescence wavelength:')
                        predcol2.markdown(f'## Predicted PLQY:')
                        predcol1.markdown(f'### {pred_lum} nm in dichloromethane')
                        predcol2.markdown(f'### {str_plqy}% in dichloromethane')
                        if pred_plqy <= 10:
                            predcol2.image('low_qy.png', width=200)
                            predcol2.markdown(f'### Low PLQY (0-10%)')
                        elif 50 >= pred_plqy > 10:
                            predcol2.image('moderate_qy.png', width=200)
                            predcol2.markdown(f'### Moderate PLQY (10-50%)')
                        else:
                            predcol2.image('high_qy.png', width=200)
                            predcol2.markdown(f'### High PLQY (50-100%)')
                        df['res_dist'] = df['L1_ecfp'].apply(lambda ecfp1: hamming_distance(L1_res_ecfp, ecfp1)) + df['L1_ecfp'].apply(lambda ecfp2: hamming_distance(L2_res_ecfp, ecfp2)) + df['L3_ecfp'].apply(lambda ecfp3: hamming_distance(L3_res_ecfp, ecfp3))
                        search_df = df[df['res_dist'] == df['res_dist'].min()]

                        st.markdown(f'### Below are shown the most similar complexes found in the IrLumDB:')
                        col1search, col2search, col3search, col4search, col5search, col6search, col7search, col8search = st.columns([1, 1, 1, 1, 1, 2, 2, 2])
                        col1search.markdown(f'**λlum,nm**')
                        col2search.markdown(f'**PLQY**')
                        col3search.markdown(f'**Solvent**')
                        col4search.markdown(f'**Abbreviation**')
                        col5search.markdown(f'**Source**')
                        col6search.markdown(f'**L1**')
                        col7search.markdown(f'**L2**')
                        col8search.markdown(f'**L3**')
                        for lam, qy, solvent, doi, abbr, L1_df, L2_df, L3_df in zip(search_df['Max_wavelength(nm)'], search_df['PLQY'], search_df['Solvent'], search_df['DOI'], search_df['Abbreviation_in_the_article'], search_df['L1'], search_df['L2'], search_df['L3']):
                            col1result, col2result, col3result, col4result, col5result, col6result, col7result, col8result = st.columns([1, 1, 1, 1, 1, 2, 2, 2])
                            col1result.markdown(f'**{lam} nm**')
                            col2result.markdown(f'**{qy}**')
                            col3result.markdown(f'**{solvent}**')
                            col4result.markdown(f'**{abbr}**')
                            col5result.markdown(f'**https://doi.org/{doi}**')
                            col6result.image(draw_molecule(L1_df), caption=L1_df)
                            col7result.image(draw_molecule(L2_df), caption=L2_df)
                            col8result.image(draw_molecule(L3_df), caption=L3_df)
                    else:
                        st.markdown(f'### Found this complex in IrLumDB:')
                        col1search, col2search, col3search, col4search, col5search = st.columns([1, 1, 1, 3, 4])
                        col1search.markdown(f'**λlum,nm**')
                        col2search.markdown(f'**PLQY**')
                        col3search.markdown(f'**Solvent:**')
                        col4search.markdown(f'**Abbreviation in the source:**')
                        col5search.markdown(f'**Source**')

                        for lam, qy, solvent, doi, abbr in zip(search_df['Max_wavelength(nm)'], search_df['PLQY'], search_df['Solvent'], search_df['DOI'], search_df['Abbreviation_in_the_article']):
                            col1result, col2result, col3result, col4result, col5result = st.columns([1, 1, 1, 3, 4])
                            col1result.markdown(f'**{lam} nm**')
                            col2result.markdown(f'**{qy}**')
                            col3result.markdown(f'**{solvent}**')
                            col4result.markdown(f'**{abbr}**')
                            col5result.markdown(f'**https://doi.org/{doi}**')

            else:
                st.error("Incorrect SMILES entered")
        else:
            st.error("Please enter all three ligands")

with tabs[2]:
    min_value = 400
    max_value = 810
    initial_value = (500, 600)

    slider_value = st.slider(
        label="λlum,nm",
        min_value=min_value,
        max_value=max_value,
        value=initial_value
    )

    sort_param = st.radio(
        "Sort data by:",
        ["PLQY", "λlum,nm"])

    if st.button("Set predicted wavelength range"):
        if sort_param == "PLQY":
            range_df = df_pred[(df_pred['pred_lum'] <= slider_value[1]) & (df_pred['pred_lum'] >= slider_value[0])].sort_values(by='pred_PLQY', ascending=False)
        else:
            range_df = df_pred[(df_pred['pred_lum'] <= slider_value[1]) & (df_pred['pred_lum'] >= slider_value[0])].sort_values(by='pred_lum', ascending=False)
        range_df = range_df[:500]
        num = str(range_df.shape[0])
        st.success(f"Selected range: {slider_value}. Found {num} entries:")
        col1range, col2range, col3range, col4range, col5range, col6range = st.columns([1, 1, 2, 2, 2, 2])
        col1range.markdown(f'**PLQY**')
        col2range.markdown(f'**λlum,nm**')
        col3range.markdown(f'**PubChem**')
        col4range.markdown(f'**L1**')
        col5range.markdown(f'**L2**')
        col6range.markdown(f'**L3**')

        for plqy, lam, cid, L1, in zip(range_df['pred_PLQY'],
                                       range_df['pred_lum'],
                                       range_df['CID'],
                                       range_df['SMILES_charge']):

            col1, col2, col3, col4, col5, col6, = st.columns([1, 1, 2, 2, 2, 2])
            plqy = plqy*100
            col1.markdown(f'**{plqy}%**')
            col2.markdown(f'**{lam}nm**')
            col3.markdown(f'**https://pubchem.ncbi.nlm.nih.gov/compound/{cid}**')
            col4.image(draw_molecule(L1), caption=L1)
            col5.image(draw_molecule(L1), caption=L1)
            col6.image(draw_molecule('CC(=O)/C=C(/C)[O-]'), caption='CC(=O)/C=C(/C)[O-]')

    inchi = st.text_input(
            "InChI",
            placeholder='InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3',
            key='InChI')
    if inchi:
        if Chem.MolFromInchi(inchi) is not None:
            smile_code = Chem.MolToSmiles(Chem.MolFromInchi(inchi))
            st.markdown(f"``{smile_code}``")
            st.image(draw_molecule(smile_code), caption=smile_code)
        else:
            st.markdown(f"**Неверный InChI**")
