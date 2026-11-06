
import streamlit as st
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, PDBList
import io
import os

st.set_page_config(page_title="Protein Contact Residue Server", layout="wide")

st.title("🧬 Protein Contact Residue Server")
st.write("Determine contact residues between two protein chains using customizable distance thresholds.")

# Operational chemical classifications of amino acids
amino_acid_groups = {
    'Hydrophobic': ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'],
    'Polar': ['SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'],
    'Positively charged': ['LYS', 'ARG', 'HIS'],
    'Negatively charged': ['ASP', 'GLU'],
    'Aromatic': ['PHE', 'TYR', 'TRP']
}

def get_amino_acid_statistics(all_unique_res):
    """Calculates residue frequencies, group distributions, and prepares dataframes for analysis."""
    res_names_only = [res[:3].upper() for res in all_unique_res if len(res) >= 3]
    tot_unique = len(res_names_only)

    if tot_unique == 0:
        return None, None, None

    # Individual amino acid stats
    aa_counts = pd.Series(res_names_only).value_counts().sort_index()
    df_aa_stats = pd.DataFrame({
        "Frequency": aa_counts,
        "Percentage (%)": (aa_counts / tot_unique * 100).round(2)
    })

    # Chemical groups mapping
    group_counts = {}
    for group_name, aa_list in amino_acid_groups.items():
        count = sum(1 for r in res_names_only if r in aa_list)
        group_counts[group_name] = count

    df_groups = pd.DataFrame({
        "Category": list(group_counts.keys()),
        "Count": list(group_counts.values()),
        "Percentage (%)": [round((c / tot_unique) * 100, 2) for c in group_counts.values()]
    })
    df_groups.index = df_groups.index + 1

    return df_aa_stats, df_groups, aa_counts

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
    models = list(structure.get_models())
    selected_model = models[0]
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
            st.subheader(f"Inter-chain Contacts: Chain {chain_a} and Chain {chain_b} at {selected_cutoff} Å")

            ch_a_obj = selected_model[chain_a]
            ch_b_obj = selected_model[chain_b]

            atom_contacts = []
            unique_pairs = {}
            interface_residues_a = set()
            interface_residues_b = set()

            for res_a in ch_a_obj.get_residues():
                if res_a.id[0] != " ":
                    continue
                for res_b in ch_b_obj.get_residues():
                    if res_b.id[0] != " ":
                        continue

                    # Scan all atom pairs to compute separate metrics
                    min_dist_for_pair = float("inf")
                    closest_atoms = (None, None)
                    pair_key = (res_a.get_resname(), res_a.id[1], res_b.get_resname(), res_b.id[1])

                    for atom_a in res_a.get_atoms():
                        for atom_b in res_b.get_atoms():
                            dist = atom_a - atom_b
                            if dist <= selected_cutoff:
                                # Record every individual atom-atom contact
                                atom_contacts.append({
                                    "Residue A": f"{res_a.get_resname()}{res_a.id[1]}",
                                    "Atom A": atom_a.get_name(),
                                    "Residue B": f"{res_b.get_resname()}{res_b.id[1]}",
                                    "Atom B": atom_b.get_name(),
                                    "Distance (Å)": round(dist, 3)
                                })
                                # Track unique residue pair relationships
                                if dist < min_dist_for_pair:
                                    min_dist_for_pair = dist
                                    closest_atoms = (atom_a.get_name(), atom_b.get_name())

                                # Add to unique interface residue sets
                                interface_residues_a.add(f"{res_a.get_resname()}{res_a.id[1]}")
                                interface_residues_b.add(f"{res_b.get_resname()}{res_b.id[1]}")

                    if min_dist_for_pair <= selected_cutoff:
                        unique_pairs[pair_key] = {
                            "Residue A Name": res_a.get_resname(),
                            "Residue A ID": res_a.id[1],
                            "Residue B Name": res_b.get_resname(),
                            "Residue B ID": res_b.id[1],
                            "Minimum Distance (Å)": round(min_dist_for_pair, 3),
                            "Closest Atoms (A-B)": f"{closest_atoms[0]}-{closest_atoms[1]}"
                        }

            # Convert results to lists/DataFrames
            df_pairs = pd.DataFrame(list(unique_pairs.values()))
            df_atoms = pd.DataFrame(atom_contacts)

            # Create DataFrame for Unique Interface Residues
            df_interface_a = pd.DataFrame({"Chain": chain_a, "Residue": sorted(list(interface_residues_a))})
            df_interface_b = pd.DataFrame({"Chain": chain_b, "Residue": sorted(list(interface_residues_b))})
            df_interface_all = pd.concat([df_interface_a, df_interface_b], ignore_index=True)

            # Adjust DataFrame Indexes to be 1-based
            if not df_pairs.empty:
                df_pairs.index = df_pairs.index + 1
            if not df_atoms.empty:
                df_atoms.index = df_atoms.index + 1
            if not df_interface_all.empty:
                df_interface_all.index = df_interface_all.index + 1

            # Display Metrics Dashboard
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Atom–Atom Contacts", len(df_atoms))
            with col2:
                st.metric("Residue–Residue Pairs", len(df_pairs))
            with col3:
                total_unique_res = len(interface_residues_a) + len(interface_residues_b)
                st.metric("Unique Interface Residues", f"{total_unique_res} (C1: {len(interface_residues_a)}, C2: {len(interface_residues_b)})")

            if not df_pairs.empty:
                tab1, tab2, tab3 = st.tabs(["Residue-Residue Pairs", "Detailed Atom-Atom Contacts", "Unique Interface Residues & Property Analysis"])

                with tab1:
                    st.dataframe(df_pairs, use_container_width=True)
                    csv_pairs = df_pairs.to_csv(index=True).encode("utf-8")
                    st.download_button(
                        label="📥 Download Residue Pairs CSV",
                        data=csv_pairs,
                        file_name=f"residue_pairs_{chain_a}_{chain_b}_{selected_cutoff}A.csv",
                        mime="text/csv"
                    )

                with tab2:
                    st.dataframe(df_atoms, use_container_width=True)
                    csv_atoms = df_atoms.to_csv(index=True).encode("utf-8")
                    st.download_button(
                        label="📥 Download Atom Contacts CSV",
                        data=csv_atoms,
                        file_name=f"atom_contacts_{chain_a}_{chain_b}_{selected_cutoff}A.csv",
                        mime="text/csv"
                    )

                with tab3:
                    st.write("**Unique interface residues per chain (no duplicates):**")
                    st.dataframe(df_interface_all, use_container_width=True)
                    csv_interface = df_interface_all.to_csv(index=True).encode("utf-8")
                    st.download_button(
                        label="📥 Download Interface Residues CSV",
                        data=csv_interface,
                        file_name=f"interface_residues_{chain_a}_{chain_b}_{selected_cutoff}A.csv",
                        mime="text/csv"
                    )

                    st.markdown("---")

                    all_unique_res = list(interface_residues_a.union(interface_residues_b))
                    df_aa_stats, df_groups, aa_counts = get_amino_acid_statistics(all_unique_res)

                    if df_aa_stats is not None:
                        # Section i: Amino acid frequency distribution (rendered vertically below)
                        st.write("**i) Amino acid frequency distribution**")
                        st.dataframe(df_aa_stats, use_container_width=True)

                        # Frequency bar chart
                        st.write("**Amino Acid Count Chart:**")
                        st.bar_chart(aa_counts)

                        st.markdown("---")

                        # Section ii: Group Distributions (rendered vertically below)
                        st.write("**ii) Group Distributions**")
                        st.dataframe(df_groups, use_container_width=True)

                        # Group bar chart
                        st.write("**Property Group Chart:**")
                        st.bar_chart(df_groups.set_index("Category")["Count"])
                    else:
                        st.write("No residues to display properties for.")
            else:
                st.info(f"No contacts found between Chain {chain_a} and Chain {chain_b} within {selected_cutoff} Å.")
else:
    st.info("Please load a structure from the sidebar to begin analyzing contact residues.")
