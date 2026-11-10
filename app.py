
import streamlit as st
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, PDBList
import io
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="Protein Contact Residue Server", layout="wide")

# Inject Custom CSS to highlight Titles, Tabs and Sidebar Headers
st.markdown("""
    <style>
    /* Highlight main title */
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #1E3A8A !important;
        text-align: center;
        margin-bottom: 5px;
        padding: 10px;
        background: linear-gradient(90deg, #EFF6FF, #DBEAFE, #EFF6FF);
        border-radius: 12px;
        border-left: 8px solid #3B82F6;
    }

    /* Highlight and style navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #F3F4F6;
        padding: 8px 12px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 6px;
        color: #4B5563 !important;
        font-weight: 600 !important;
        padding: 0 16px;
        border: 1px solid #E5E7EB;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #2563EB !important;
        border-color: #3B82F6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: 1px solid #2563EB !important;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2);
    }

    /* Styled Custom Headers for Sidebar & Main Panel Widgets */
    .sidebar-header-highlight {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        background-color: #1E3A8A !important;
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 20px;
        margin-bottom: 12px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Display Highlighted Title
st.markdown('<div class="main-title">⚕️ Protein Contact Residue Server</div>', unsafe_allow_html=True)
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

# Sidebar with styled headers
st.sidebar.markdown('<div class="sidebar-header-highlight">1. Load PDB Structure</div>', unsafe_allow_html=True)
input_method = st.sidebar.radio("Choose Source:", ("Fetch from PDB (RCSB)", "Upload PDB File"))

parser = PDBParser(QUIET=True)
structure = None
pdb_raw_data = None

if input_method == "Fetch from PDB (RCSB)":
    pdb_id = st.sidebar.text_input("Enter 4-letter PDB ID (e.g., 6Q9O):", "6Q9O").strip()
    if pdb_id:
        try:
            pdbl = PDBList()
            pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="pdb")
            with open(pdb_filename, "r") as f:
                pdb_raw_data = f.read()
            structure = parser.get_structure(pdb_id, pdb_filename)
            st.success(f"Successfully loaded structure for PDB ID: {pdb_id.upper()}")
        except Exception as e:
            st.error(f"Error fetching PDB ID {pdb_id}: {e}")

else:
    uploaded_file = st.sidebar.file_uploader("Upload a .pdb file:", type=["pdb"])
    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.getvalue()
            pdb_raw_data = raw_bytes.decode("utf-8")
            stringio = io.StringIO(pdb_raw_data)
            structure = parser.get_structure("uploaded_structure", stringio)
            st.success("Successfully loaded uploaded PDB file.")
        except Exception as e:
            st.error(f"Error parsing uploaded PDB file: {e}")

if structure:
    models = list(structure.get_models())
    selected_model = models[0]
    all_chains = [chain.id for chain in selected_model.get_chains()]

    st.sidebar.markdown('<div class="sidebar-header-highlight">2. Parameters</div>', unsafe_allow_html=True)
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
            ch_a_obj = selected_model[chain_a]
            ch_b_obj = selected_model[chain_b]

            atom_contacts = []
            unique_pairs = {}
            interface_residues_a = set()
            interface_residues_b = set()
            contact_links_visualization = []

            for res_a in ch_a_obj.get_residues():
                if res_a.id[0] != " ":
                    continue
                for res_b in ch_b_obj.get_residues():
                    if res_b.id[0] != " ":
                        continue

                    min_dist_for_pair = float("inf")
                    closest_atoms = (None, None)
                    pair_key = (res_a.get_resname(), res_a.id[1], res_b.get_resname(), res_b.id[1])
                    visual_atom_pair = None

                    for atom_a in res_a.get_atoms():
                        for atom_b in res_b.get_atoms():
                            dist = atom_a - atom_b
                            if dist <= selected_cutoff:
                                atom_contacts.append({
                                    "Residue A": f"{res_a.get_resname()}{res_a.id[1]}",
                                    "Atom A": atom_a.get_name(),
                                    "Residue B": f"{res_b.get_resname()}{res_b.id[1]}",
                                    "Atom B": atom_b.get_name(),
                                    "Distance (Å)": round(dist, 3)
                                })
                                if dist < min_dist_for_pair:
                                    min_dist_for_pair = dist
                                    closest_atoms = (atom_a.get_name(), atom_b.get_name())
                                    visual_atom_pair = (atom_a.get_coord(), atom_b.get_coord())

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
                        if visual_atom_pair is not None:
                            contact_links_visualization.append({
                                "pair_label": f"{res_a.get_resname()}{res_a.id[1]} - {res_b.get_resname()}{res_b.id[1]}",
                                "coord_a": visual_atom_pair[0],
                                "coord_b": visual_atom_pair[1],
                                "label_a": f"{res_a.get_resname()}{res_a.id[1]}",
                                "label_b": f"{res_b.get_resname()}{res_b.id[1]}",
                                "spec_a": {"chain": chain_a, "resi": res_a.id[1]},
                                "spec_b": {"chain": chain_b, "resi": res_b.id[1]}
                            })

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

            # Create side-by-side Layout: Left side for 3D molecular view, Right side for filter controls
            col_visualizer, col_filters = st.columns([3, 1])

            with col_filters:
                st.markdown('<div class="sidebar-header-highlight">3. 3D Filter Display</div>', unsafe_allow_html=True)
                enable_labels = st.checkbox("Enable Residue Labels", value=True)
                if contact_links_visualization:
                    available_pairs = [link["pair_label"] for link in contact_links_visualization]
                    selected_visual_pairs = st.multiselect(
                        "Select contact residue pairs to display (leave empty to show all):",
                        options=available_pairs,
                        default=[]
                    )
                    # Filter links based on selection
                    if selected_visual_pairs:
                        filtered_links = [link for link in contact_links_visualization if link["pair_label"] in selected_visual_pairs]
                    else:
                        filtered_links = contact_links_visualization
                else:
                    st.info("No contacts available to filter.")
                    filtered_links = []

            with col_visualizer:
                st.subheader(f"🔼 3D Molecular View: Chain {chain_a} and Chain {chain_b}")
                st.write("Interactive 3D representation showing contacting residues linked together.")

                if pdb_raw_data:
                    try:
                        # Format molecular instructions using direct py3Dmol/javascript scripting wrapper
                        escaped_pdb = pdb_raw_data.replace("\n", "\\n").replace("\"", "\\\"").replace("\'", "\\\'")

                        cylinder_lines_js = ""
                        labels_js = ""
                        sidechains_js = ""

                        for link in filtered_links:
                            ca = link["coord_a"]
                            cb = link["coord_b"]
                            # Append cylinders (dashed)
                            cylinder_lines_js += f"""
                            viewer.addCylinder({{
                                start: {{x: {ca[0]}, y: {ca[1]}, z: {ca[2]}}},
                                end: {{x: {cb[0]}, y: {cb[1]}, z: {cb[2]}}},
                                radius: 0.15,
                                fromCap: 1,
                                toCap: 1,
                                color: 'yellow',
                                dashed: true
                            }});
                            """
                            # Labels rendering with transparent background
                            if enable_labels:
                                labels_js += f"""
                                viewer.addLabel('{link["label_a"]}', {{fontColor:'blue', backgroundColor:'#DCE7F3', fontSize:10, backgroundOpacity:0.0, borderThickness:0.0}}, {{chain: '{link["spec_a"]["chain"]}', resi: {link["spec_a"]["resi"]}}});
                                viewer.addLabel('{link["label_b"]}', {{fontColor:'red', backgroundColor:'#FEE2E2', fontSize:10, backgroundOpacity:0.0, borderThickness:0.0}}, {{chain: '{link["spec_b"]["chain"]}', resi: {link["spec_b"]["resi"]}}});
                                """
                            # Sidechains (Stick styling)
                            sidechains_js += f"""
                            viewer.setStyle({{chain: '{link["spec_a"]["chain"]}', resi: {link["spec_a"]["resi"]}}}, {{stick: {{colorspec: 'cyan'}}}});
                            viewer.setStyle({{chain: '{link["spec_b"]["chain"]}', resi: {link["spec_b"]["resi"]}}}, {{stick: {{colorspec: 'orange'}}}});
                            """

                        py3dmol_html = f"""
                        <div id="3dmol_canvas" style="width: 100%; height: 500px;"></div>
                        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/1.7.0/3Dmol-min.js"></script>
                        <script>
                            var element = document.getElementById('3dmol_canvas');
                            var config = {{ backgroundColor: 'black' }};
                            var viewer = $3Dmol.createViewer(element, config);
                            viewer.addModel('{escaped_pdb}', 'pdb');

                            // Clear styling of all molecules/chains
                            viewer.setStyle({{}}, {{}});

                            // Display ONLY the selected interacting chains
                            viewer.setStyle({{chain: '{chain_a}'}}, {{cartoon: {{color: '#3B82F6'}}}});
                            viewer.setStyle({{chain: '{chain_b}'}}, {{cartoon: {{color: '#EF4444'}}}});

                            {cylinder_lines_js}
                            {sidechains_js}
                            {labels_js}
                            viewer.zoomTo({{chain: ['{chain_a}', '{chain_b}']}});
                            viewer.render();
                        </script>
                        """
                        components.html(py3dmol_html, height=510)
                    except Exception as e:
                        st.error(f"Could not render 3D structural model: {e}")

            # --- Inter-chain Contacts Results Table Header ---
            st.subheader(f"Inter-chain Contacts: Chain {chain_a} and Chain {chain_b} at {selected_cutoff} Å")

            # Display Metrics Dashboard rearranged: 1. Unique Interface Residues, 2. Residue-Residue Pairs, 3. Atom-Atom Contacts
            col1, col2, col3 = st.columns(3)
            with col1:
                total_unique_res = len(interface_residues_a) + len(interface_residues_b)
                st.metric("Unique Interface Residues", f"{total_unique_res} (C1: {len(interface_residues_a)}, C2: {len(interface_residues_b)})")
            with col2:
                st.metric("Residue–Residue Pairs", len(df_pairs))
            with col3:
                st.metric("Atom–Atom Contacts", len(df_atoms))

            if not df_pairs.empty:
                # Reordered tabs: 1. Unique Interface, 2. Residue-Residue Pairs, 3. Detailed Atom-Atom
                tab1, tab2, tab3 = st.tabs([
                    "🧬 Unique Interface Residues & Property Analysis",
                    "🔗 Residue-Residue Pairs",
                    "🔍 Detailed Atom-Atom Contacts"
                ])

                with tab1:
                    st.write("**Unique interface residues per chain (no duplicates):**")
                    st.dataframe(df_interface_all, use_container_width=True)
                    csv_interface = df_interface_all.to_csv(index=True).encode("utf-8")
                    st.download_button(
                        label="▄ Download Interface Residues CSV",
                        data=csv_interface,
                        file_name=f"interface_residues_{chain_a}_{chain_b}_{selected_cutoff}A.csv",
                        mime="text/csv"
                    )

                    st.markdown("---")

                    all_unique_res = list(interface_residues_a.union(interface_residues_b))
                    df_aa_stats, df_groups, aa_counts = get_amino_acid_statistics(all_unique_res)

                    if df_aa_stats is not None:
                        # Section i: Amino acid frequency distribution with side-by-side bar chart
                        st.write("**i) Amino acid frequency distribution**")
                        col_table_i, col_chart_i = st.columns(2)
                        with col_table_i:
                            st.dataframe(df_aa_stats, use_container_width=True)
                            csv_aa_stats = df_aa_stats.to_csv(index=True).encode("utf-8")
                            st.download_button(
                                label="▄ Download Amino Acid Frequency CSV",
                                data=csv_aa_stats,
                                file_name=f"amino_acid_frequency_{chain_a}_{chain_b}_{selected_cutoff}A.csv",
                                mime="text/csv"
                            )
                        with col_chart_i:
                            st.bar_chart(aa_counts, use_container_width=True)

                        st.markdown("---")

                        # Section ii: Group Distributions with side-by-side bar chart
                        st.write("**ii) Group Distributions**")
                        col_table_ii, col_chart_ii = st.columns(2)
                        with col_table_ii:
                            st.dataframe(df_groups, use_container_width=True)
                            csv_groups = df_groups.to_csv(index=True).encode("utf-8")
                            st.download_button(
                                label="▄ Download Group Distribution CSV",
                                data=csv_groups,
                                file_name=f"amino_acid_groups_{chain_a}_{chain_b}_{selected_cutoff}A.csv",
                                mime="text/csv"
                            )
                        with col_chart_ii:
                            st.bar_chart(df_groups.set_index("Category")["Count"], use_container_width=True)
                    else:
                        st.write("No residues to display properties for.")

                with tab2:
                    st.dataframe(df_pairs, use_container_width=True)
                    csv_pairs = df_pairs.to_csv(index=True).encode("utf-8")
                    st.download_button(
                        label="▄ Download Residue Pairs CSV",
                        data=csv_pairs,
                        file_name=f"residue_pairs_{chain_a}_{chain_b}_{selected_cutoff}A.csv",
                        mime="text/csv"
                    )

                with tab3:
                    st.dataframe(df_atoms, use_container_width=True)
                    csv_atoms = df_atoms.to_csv(index=True).encode("utf-8")
                    st.download_button(
                        label="▄ Download Atom Contacts CSV",
                        data=csv_atoms,
                        file_name=f"atom_contacts_{chain_a}_{chain_b}_{selected_cutoff}A.csv",
                        mime="text/csv"
                    )
            else:
                st.info(f"No contacts found between Chain {chain_a} and Chain {chain_b} within {selected_cutoff} Å.")
else:
    st.info("Please load a structure from the sidebar to begin analyzing contact residues.")
