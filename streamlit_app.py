import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import pickle
from xgboost import XGBRegressor
from streamlit_ketcher import st_ketcher
from molfeat.calc import FPCalculator

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.Draw.MolToImage(mol)

def canonize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

calc = FPCalculator("ecfp")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.set_page_config(page_title='IrLumDB', layout="wide")

df = pd.read_csv('BigIrDB_v17.csv')
df['L1'] = df['L1'].apply(lambda x: canonize_smiles(x))
df['L2'] = df['L2'].apply(lambda x: canonize_smiles(x))
df['L3'] = df['L3'].apply(lambda x: canonize_smiles(x))
lum = df[['λlum,nm', 'QY', 'solvent', 'DOI', 'ZEROS']]
lum = lum[lum['ZEROS'] != 0]
lum = lum[~lum['QY'].isna()]
lum['QY'] = lum['QY'].apply(lambda x: float(x.replace('<', '').replace(',','.')))
lum = lum[lum['solvent'].apply(lambda x: x in ['CH2Cl2', 'CH3CN', 'toluene', 'CH3OH', 'THF'])]

col1intro, col2intro = st.columns([2, 1])
col1intro.markdown("""
# IrLumDB App v1.0

The ”IrLumDB App” is an ML-based service integrated with the experimental database to predict luminescence wavelength of bis-cyclometalated iridium(III) complexes requiring only molecular formula of the ligands as a feature.

### There are currently two operation modes:
* exploration of the database (**“explore”** window)
* prediction of luminescence wavelength (**“search and predict”** window)
""")

col2intro.image('TOC.png')

tabs = st.tabs(["Explore", "Search and Predict"])

with tabs[0]:
    fig_lum = px.scatter(lum, x="λlum,nm", y="QY", color="solvent", hover_data={'DOI': True}, title='Space of photophysical properties for bis-cyclometalated iridium(III) complexes')
    fig_lum.update_layout(yaxis_title='PLQY')
    st.plotly_chart(fig_lum)

    fig_qy = px.histogram(lum, x='QY', nbins=64, title='PLQY distribution in the IrLumDB')
    fig_qy.update_layout(yaxis_title='Number of entries')
    fig_qy.update_layout(xaxis_title='PLQY')
    st.plotly_chart(fig_qy)

    fig = px.histogram(df, x='λlum,nm', nbins=64, title='Maximum wavelength(nm) distribution in the IrLumDB')
    fig.update_layout(yaxis_title='Number of entries')
    st.plotly_chart(fig)

    st.markdown('The “IrLumDB” database contains data about **1454** experimentally measured luminescence spectra of **1287** unique iridium(III) complexes reported in the **340** literature papers. To explore the database, please choose the desired emission wavelength interval below:')
    min_value = df['λlum,nm'].min()
    max_value = df['λlum,nm'].max()
    initial_value = (400, 500)
    max_interval_length = 10

    slider_value = st.slider(
        "",
        min_value=min_value,
        max_value=max_value,
        value=initial_value
    )

    if st.button("Set wavelength range"):
        range_df = df[(df['λlum,nm'] <= slider_value[1]) & (df['λlum,nm'] >= slider_value[0])].sort_values(by='λlum,nm')
        num = str(range_df.shape[0])
        st.success(f"Selected range: {slider_value}. Found {num} entries:")
        col1range, col2range, col3range, col4range, col5range, col6range, col7range = st.columns([1, 1, 2, 2, 2, 2, 2])
        col1range.markdown(f'**λlum,nm**')
        col2range.markdown(f'**Solvent:**')
        col3range.markdown(f'**Abbreviation in the source:**')
        col4range.markdown(f'**Source**')
        col5range.markdown(f'**L1**')
        col6range.markdown(f'**L2**')
        col7range.markdown(f'**L3**')

        for lam, solvent, doi, abbr, L1, L2, L3 in zip(range_df['λlum,nm'],
                                                       range_df['solvent'],
                                                       range_df['DOI'],
                                                       range_df['Abbreviation_in_the_article'],
                                                       range_df['L1'],
                                                       range_df['L2'],
                                                       range_df['L3']):

            col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 2, 2, 2, 2, 2])
            col1.markdown(f'**{lam} nm**')
            col2.markdown(f'**{solvent}**')
            col3.markdown(f'**{abbr}**')
            col4.markdown(f'**https://doi.org/{doi}**')
            col5.image(draw_molecule(L1), caption=L1)
            col6.image(draw_molecule(L2), caption=L2)
            col7.image(draw_molecule(L3), caption=L3)

with tabs[1]:

    st.markdown("""Please enter SMILES of the ligands (or draw the structural formula in the corresponding window) and press “Search in the database and predict maximum wavelength (nm)” button to perform the prediction. If the complex exists in the database, experimental data will be displayed. If the complex does not exist in the database, the predicted luminescence wavelength will appear.

Usage notes:
* The desired complexes usually contain two cyclometalated ligands and one ancillary ligand; thus L1 and L2 should correspond to the cyclometalated ligands and L3 should correspond to the ancillary ligand.
* Some ligands make formally covalent bonds with the Ir(III) ion. For these a negatively charged bond-forming atom should be drawn in the SMILES of corresponding ligand.
* The ML model uses only spectroscopic data obtained in **dichloromethane solvent**, thus the predicted luminescence wavelength is aimed to be also in dichloromethane solution of the corresponding complex.

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
                canonize_l1 = Chem.MolToSmiles(mol1)
                canonize_l2 = Chem.MolToSmiles(mol2)
                canonize_l3 = Chem.MolToSmiles(mol3)
                col1.image(draw_molecule(L1), caption=L1)
                col2.image(draw_molecule(L2), caption=L2)
                col3.image(draw_molecule(L3), caption=L3)
                search_df = df[(df['L1'] == canonize_l1) & (df['L2'] == canonize_l2) & (df['L3'] == canonize_l3)]
                if search_df.shape[0] == 0:
                    L_res = calc(mol1) + calc(mol2) + calc(mol3)
                    L_res = L_res.reshape(1, -1)
                    pred_lum = str(int(round(model_lum.predict(L_res)[0], 0)))
                    pred_plqy = str(round(model_plqy.predict(L_res)[0], 3))
                    predcol1, predcol2 = st.columns(2)
                    predcol1.markdown(f'## Predicted luminescence wavelength:')
                    predcol2.markdown(f'## Predicted PLQY:')
                    predcol1.markdown(f'### {pred_lum} nm in dichloromethane')
                    predcol2.markdown(f'### {pred_plqy} in dichloromethane')
                else:
                    st.markdown(f'### Found this complex in IrLumDB:')
                    col1search, col2search, col3search, col4search, col5search = st.columns([1, 1, 1, 3, 4])
                    col1search.markdown(f'**λlum,nm**')
                    col2search.markdown(f'**PLQY**')
                    col3search.markdown(f'**Solvent:**')
                    col4search.markdown(f'**Abbreviation in the source:**')
                    col5search.markdown(f'**Source**')
                    for lam, qy, solvent, doi, abbr in zip(search_df['λlum,nm'], search_df['QY'], search_df['solvent'], search_df['DOI'], search_df['Abbreviation_in_the_article']):
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
