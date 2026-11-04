
import streamlit as st
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, PDBList
import io
import os

st.set_page_config(page_title="Protein Contact Residue Server", layout="wide")

st.title("🧬 Protein Contact Residue Server")
st.write("Determine contact residues between two protein chains using customizable distance thresholds.")

# Sidebar for structure input
st.sidebar.header("1. Load PDB Structure")
input_method = st.sidebar.radio("Choose Source:", ("Fetch from PDB (RCSB)", "Upload PDB File"))

parser = PDBParser(QUIET=True)
structure = None

if input_method == "Fetch from PDB (RCSB)":
    pdb_id = st.sidebar.text_input("Enter 4-letter PDB ID (e.g., 1A8O):", "1A8O").strip()
    if pdb_id:
        try:
            pdbl = PDBList()
            # Fetch file and load
            pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="pdb")
            structure = parser.get_structure(pdb_id, pdb_filename)
            st.success(f"Successfully loaded structure for PDB ID: {pdb_id.upper()}")
        except Exception as e:
            st.error(f"Error fetching PDB ID {pdb_id}: {e}")

else:
    uploaded_file = st.sidebar.file_uploader("Upload a .pdb file:", type=["pdb"])
    if uploaded_file is not None:
        try:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            structure = parser.get_structure("uploaded_structure", stringio)
            st.success("Successfully loaded uploaded PDB file.")
        except Exception as e:
            st.error(f"Error parsing uploaded PDB file: {e}")

if structure:
    # Get models and chain list
    models = list(structure.get_models())
    selected_model = models[0]  # Focus on the first model
    
    all_chains = [chain.id for chain in selected_model.get_chains()]
    
    st.sidebar.header("2. Parameters")
    if len(all_chains) >= 2:
        chain_a = st.sidebar.selectbox("Select Chain 1:", all_chains, index=0)
        chain_b = st.sidebar.selectbox("Select Chain 2:", all_chains, index=min(1, len(all_chains)-1))
    else:
        st.sidebar.warning("The loaded structure has fewer than 2 chains.")
        chain_a = all_chains[0] if all_chains else None
        chain_b = None

    cutoff_options = {
        4.0: "4 Å (Identify close contacts)",
        5.0: "5 Å (Primary contact definition)",
        6.0: "6 Å (Identify broader interfaces)"
    }
    
    selected_cutoff = st.sidebar.selectbox(
        "Select Distance Cut-off:",
        options=list(cutoff_options.keys()),
        format_func=lambda x: cutoff_options[x],
        index=1
    )

    if chain_a and chain_b:
        if chain_a == chain_b:
            st.warning("Please select two different chains for inter-chain contact analysis.")
        else:
            st.subheader(f"Inter-chain Contacts between Chain **{chain_a}** and Chain **{chain_b}** at **{selected_cutoff} Å**")
            
            ch_a_obj = selected_model[chain_a]
            ch_b_obj = selected_model[chain_b]
            
            contacts = []
            
            # Compare heavy atoms of selected chains
            for res_a in ch_a_obj.get_residues():
                # Skip heteroatoms / water if necessary
                if res_a.id[0] != " ":
                    continue
                for res_b in ch_b_obj.get_residues():
                    if res_b.id[0] != " ":
                        continue
                    
                    # Check distance between any atom of res_a and any atom of res_b
                    min_dist = float('inf')
                    closest_atoms = (None, None)
                    for atom_a in res_a.get_atoms():
                        for atom_b in res_b.get_atoms():
                            dist = atom_a - atom_b
                            if dist < min_dist:
                                min_dist = dist
                                closest_atoms = (atom_a.get_name(), atom_b.get_name())
                    
                    if min_dist <= selected_cutoff:
                        contacts.append({
                            "Residue A Name": res_a.get_resname(),
                            "Residue A ID": res_a.id[1],
                            "Residue B Name": res_b.get_resname(),
                            "Residue B ID": res_b.id[1],
                            "Minimum Distance (Å)": round(min_dist, 3),
                            "Closest Atoms (A-B)": f"{closest_atoms[0]}-{closest_atoms[1]}"
                        })
            
            if contacts:
                df_contacts = pd.DataFrame(contacts)
                st.write(f"Found **{len(df_contacts)}** contacting residue pairs.")
                
                # Show results in a dataframe
                st.dataframe(df_contacts, use_container_width=True)
                
                # Download button for CSV data
                csv = df_contacts.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Contacts CSV",
                    data=csv,
                    file_name=f"contacts_{chain_a}_{chain_b}_{selected_cutoff}A.csv",
                    mime="text/csv"
                )
            else:
                st.info(f"No contact residues found between Chain {chain_a} and Chain {chain_b} within {selected_cutoff} Å.")
else:
    st.info("Please load a structure from the sidebar to begin analyzing contact residues.")
