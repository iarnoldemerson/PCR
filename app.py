
import streamlit as st
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, PDBList
from Bio.PDB.SASA import ShrakeRupley
import io
import os
import base64
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

st.set_page_config(page_title="Protein Contact Residue Server", layout="wide")

# Inject Custom CSS to highlight Titles, Tabs, Sidebar Headers and Justify Paragraphs
try:
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

        /* Justified text styling */
        .justified-para {
            text-align: justify !important;
            text-justify: inter-word !important;
            line-height: 1.6 !important;
            margin-bottom: 15px;
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
except Exception:
    pass

# Display Highlighted Title
try:
    st.markdown("<div class=\"main-title\">☣  Protein Contact Residue Server</div>", unsafe_allow_html=True)
except Exception:
    pass

# Introductory description paragraph describing the application
st.markdown("""<div class="justified-para">
The Protein Contact Residue Server is an advanced interactive web tool for structural biologists and biochemists. It's designed to carry out very precise distance-based calculations of physical contacts and detailed surface area computations across macromolecular binding interfaces. It can take coordinate files directly from the PDB (RCSB PDB) or from your own local files, identify inter-chain residue networks, analyze Solvent Accessible Surface Area (SASA) modifications, and classify biologically relevant interfaces.
</div>""", unsafe_allow_html=True)

# Display attached user image
image_path = "/content/Screenshot 2026-11-11 at 12.03.52…PM.png"
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True, caption="Protein Contact Residue Server Overview")

# Create Main Tabs including the requested 2D Viewer Tab
main_tab_analysis, main_tab_2d_viewer, main_tab_help = st.tabs([
    "  Interface Analysis & 3D Viewer",
    "▧ Interface Analysis & 2D Viewer",
    "❓ Help Guide"
])

# Operational chemical classifications of amino acids
amino_acid_groups = {
    'Hydrophobic': ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'],
    'Polar': ['SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'],
    'Positively charged': ['LYS', 'ARG', 'HIS'],
    'Negatively charged': ['ASP', 'GLU'],
    'Aromatic': ['PHE', 'TYR', 'TRP']
}

# Standard list of the 20 canonical amino acids
standard_20_amino_acids = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

def get_amino_acid_statistics(all_unique_res):
    """Calculates residue frequencies, group distributions, and prepares dataframes for analysis."""
    res_names_only = [res[:3].upper() for res in all_unique_res if len(res) >= 3]
    tot_unique = len(res_names_only)

    # Individual amino acid stats
    counts_dict = {aa: 0 for aa in standard_20_amino_acids}
    for r in res_names_only:
        if r in counts_dict:
            counts_dict[r] += 1

    aa_counts = pd.Series(counts_dict).sort_index()
    df_aa_stats = pd.DataFrame({
        "Frequency": aa_counts,
        "Percentage (%)": ((aa_counts / tot_unique * 100) if tot_unique > 0 else 0.0).round(2)
    })

    # Chemical groups mapping
    group_counts = {}
    for group_name, aa_list in amino_acid_groups.items():
        count = sum(1 for r in res_names_only if r in aa_list)
        group_counts[group_name] = count

    df_groups = pd.DataFrame({
        "Category": list(group_counts.keys()),
        "Count": list(group_counts.values()),
        "Percentage (%)": [round((c / tot_unique) * 100, 2) if tot_unique > 0 else 0.0 for c in group_counts.values()]
    })
    df_groups.index = df_groups.index + 1

    return df_aa_stats, df_groups, aa_counts

# Sidebar with styled headers
st.sidebar.markdown("<div class=\"sidebar-header-highlight\">1. Load PDB Structure</div>", unsafe_allow_html=True)
input_method = st.sidebar.radio("Choose Source:", ("Fetch from PDB (RCSB)", "Upload PDB File"))

parser = PDBParser(QUIET=True)
structure = None
pdb_raw_data = None
pdb_id = ""

# Initialize session state to persist loaded structure
if "pdb_raw_data" not in st.session_state:
    st.session_state.pdb_raw_data = None
if "structure" not in st.session_state:
    st.session_state.structure = None
if "pdb_id" not in st.session_state:
    st.session_state.pdb_id = ""

if input_method == "Fetch from PDB (RCSB)":
    with st.sidebar.form(key="pdb_fetch_form"):
        pdb_input_id = st.text_input("Enter 4-letter PDB ID (e.g., 6Q9O):", "").strip()
        submit_button = st.form_submit_button(label="Submit PDB ID")

    if submit_button and pdb_input_id:
        try:
            pdbl = PDBList()
            pdb_filename = pdbl.retrieve_pdb_file(pdb_input_id, pdir=".", file_format="pdb")
            with open(pdb_filename, "r") as f:
                loaded_raw_data = f.read()
            loaded_structure = parser.get_structure(pdb_input_id, pdb_filename)

            st.session_state.pdb_raw_data = loaded_raw_data
            st.session_state.structure = loaded_structure
            st.session_state.pdb_id = pdb_input_id
            st.sidebar.success(f"Successfully loaded structure for PDB ID: {pdb_input_id.upper()}")
        except Exception as e:
            st.sidebar.error(f"Error fetching PDB ID {pdb_input_id}: {e}")

    pdb_raw_data = st.session_state.pdb_raw_data
    structure = st.session_state.structure
    pdb_id = st.session_state.pdb_id

else:
    uploaded_file = st.sidebar.file_uploader("Upload a .pdb file:", type=["pdb"])
    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.getvalue()
            pdb_raw_data = raw_bytes.decode("utf-8")
            stringio = io.StringIO(pdb_raw_data)
            structure = parser.get_structure("uploaded_structure", stringio)
            st.sidebar.success("Successfully loaded uploaded PDB file.")
            pdb_id = "Uploaded"
        except Exception as e:
            st.sidebar.error(f"Error parsing uploaded PDB file: {e}")

# Main Tab 3: Help Guide
with main_tab_help:
    st.markdown("###   Structural Interface Analysis Guide")
    st.write("Welcome to the reference guide and operating manual for this application. Below are instructions for executing contact calculations, interpreting SASA properties, and downloading reports.")

    st.markdown("####   Step-by-Step Instructions")
    st.markdown("""
    1. **Load Your Protein Structure**:
       * Select either **Fetch from PDB (RCSB)** or choose **Upload PDB File**.
    2. **Configure Your Chains and Threshold**:
       * Select **Chain 1** and **Chain 2** from the dropdowns.
       * Set your desired **Distance Cut-off** (4.0 Å, 5.0 Å, or 6.0 Å).
    3. **Evaluate Surface Energy & Buried Surface Area (BSA)**:
       * **Buried Surface Area (BSA) Calculation**: calculated as $BSA = (SASA_{Unbound, A} + SASA_{Unbound, B}) - SASA_{Complex}$.
       * **Interface Size Classification**: BSA values $> 1,000\text{ Å}^2$ signify biologically relevant dimers; values $< 800\text{ Å}^2$ point to non-specific crystal packing.
       * **Interface Residues**: Residues with a decrease in SASA upon complex formation ($\Delta SASA > 0$) are highlighted.
    """)

    st.markdown("####   3D/2D Filter Display Controls")
    st.markdown("""
    The **3D/2D Filter Display** panel allows you to customize and focus on specific structural elements inside the 3D canvas:
    * **Enable Residue Labels**: Toggles text annotations indicating residue name and sequence position on top of contacting side-chains.
    * **Transparent molecular surface**: Renders a space-filling Van der Waals (VDW) style envelope over the atoms, allowing you to gauge shape complementarity.
    * **Surface Opacity**: Controls VDW envelope transparency when active.
    * **Select contact residue pairs**: A multi-select tool allowing you to focus on specific interacting partner pairs (e.g., `PHE42 - LYS12`) instead of viewing all overlapping bonds simultaneously.
    * **Custom Chain Coloring & Backgrounds**: Alters the main ribbon styles and viewport background dynamically to prepare beautiful figures for publication.
    """)

    st.markdown("####   Understanding the Analysis Tables")
    st.markdown("""
    ##### **i) Unique Interface Residues & Property Analysis**
    This dataset isolates individual amino acids that lose solvent accessibility ($\Delta SASA > 0$) upon complex formation. This panel shows:
    * **Unbound and Bound SASA**: Quantifies the transition from completely isolated monomers to the fully bound biological interface.
    * **Physicochemical Group Frequencies**: Automatically categorizes residues into operational categories (*Hydrophobic*, *Polar*, *Aromatic*, *Positively charged*, and *Negatively charged*) to identify the major electrostatic or hydrophobic drivers stabilizing the complex.

    ##### **ii) Residue-Residue Pairs**
    Displays a unique pair-wise interaction registry. If Residue A from Chain 1 makes physical contact with Residue B from Chain 2, they form a registered interactive pair. It provides the **minimum distance** observed between any structural heavy-atoms in the pair.

    ##### **iii) Detailed Atom-Atom Contacts**
    A fine-grained, high-resolution atom-level list tracking every individual heavy-atom pair (e.g., nitrogen, oxygen, carbon) that falls beneath the configured cut-off boundary. Use this table to map salt bridges, specific hydrogen bonds, and close atomic clashes.
    """)

# Initialize analysis containers to share analysis output across both tabs
analysis_results = {}

if structure:
    models = list(structure.get_models())
    selected_model = models[0]
    all_chains = [chain.id for chain in selected_model.get_chains()]

    st.sidebar.markdown("<div class=\"sidebar-header-highlight\">2. Parameters</div>", unsafe_allow_html=True)
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

    if chain_a and chain_b and chain_a != chain_b:
        ch_a_obj = selected_model[chain_a]
        ch_b_obj = selected_model[chain_b]

        sasa_data_available = False
        monomer_a_sasa = {}
        monomer_b_sasa = {}
        complex_sasa = {}
        total_excluded_sasa = 0.0
        interface_classification = "N/A"
        all_residue_sasa_records = []

        try:
            sr = ShrakeRupley()
            from copy import deepcopy
            monomer_a_model = deepcopy(ch_a_obj)
            monomer_b_model = deepcopy(ch_b_obj)

            sr.compute(monomer_a_model, level="R")
            for res in monomer_a_model.get_residues():
                if res.id[0] == " ":
                    monomer_a_sasa[res.id[1]] = res.sasa if hasattr(res, 'sasa') else 0.0

            sr.compute(monomer_b_model, level="R")
            for res in monomer_b_model.get_residues():
                if res.id[0] == " ":
                    monomer_b_sasa[res.id[1]] = res.sasa if hasattr(res, 'sasa') else 0.0

            sr.compute(selected_model, level="R")
            for chain_obj in [ch_a_obj, ch_b_obj]:
                for res in chain_obj.get_residues():
                    if res.id[0] == " ":
                        complex_sasa[(chain_obj.id, res.id[1])] = res.sasa if hasattr(res, 'sasa') else 0.0

            total_unbound_sasa_a = sum(monomer_a_sasa.values())
            total_unbound_sasa_b = sum(monomer_b_sasa.values())
            total_complex_sasa = sum([complex_sasa.get((chain_id, res_id), 0.0) for (chain_id, res_id) in complex_sasa if chain_id in [chain_a, chain_b]])

            total_excluded_sasa = (total_unbound_sasa_a + total_unbound_sasa_b) - total_complex_sasa

            if total_excluded_sasa > 1000:
                interface_classification = "Biologically Relevant Dimer (BSA > 1000 Å²)"
            elif total_excluded_sasa < 800:
                interface_classification = "Non-specific Crystal Packing (BSA < 800 Å²)"
            else:
                interface_classification = "Intermediate Contact Region (800 - 1000 Å²)"

            for chain_id, ch_obj, monomer_sasa in [(chain_a, ch_a_obj, monomer_a_sasa), (chain_b, ch_b_obj, monomer_b_sasa)]:
                for res in ch_obj.get_residues():
                    if res.id[0] != " ":
                        continue
                    res_id_val = res.id[1]
                    res_name_full = f" {res.get_resname()}{res_id_val}"
                    unbound_val = monomer_sasa.get(res_id_val, 0.0)
                    bound_val = complex_sasa.get((chain_id, res_id_val), 0.0)
                    delta_sasa = max(0.0, unbound_val - bound_val)
                    all_residue_sasa_records.append({
                        "Chain": chain_id,
                        "Residue": res_name_full.strip(),
                        "Unbound SASA (Å²)": round(unbound_val, 2),
                        "Bound SASA (Å²)": round(bound_val, 2),
                        "ΔSASA (Å²)": round(delta_sasa, 2)
                    })
            sasa_data_available = True
        except Exception as sasa_err:
            st.sidebar.error(f"SASA Calculation Error: {sasa_err}")

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
                        "Closest Atoms (A-B)": f"{(closest_atoms[0] or '')}-{(closest_atoms[1] or '')}"
                    }
                    if visual_atom_pair is not None:
                        contact_links_visualization.append({
                            "pair_label": f"{res_a.get_resname()}{res_a.id[1]} - {res_b.get_resname()}{res_b.id[1]}",
                            "coord_a": visual_atom_pair[0],
                            "coord_b": visual_atom_pair[1],
                            "label_a": f" {res_a.get_resname()}{res_a.id[1]}",
                            "label_b": f" {res_b.get_resname()}{res_b.id[1]}",
                            "spec_a": { "chain": chain_a, "resi": res_a.id[1] },
                            "spec_b": { "chain": chain_b, "resi": res_b.id[1] }
                        })

        df_pairs = pd.DataFrame(list(unique_pairs.values()))
        df_atoms = pd.DataFrame(atom_contacts)
        df_interface_all_data = [rec for rec in all_residue_sasa_records if rec["ΔSASA (Å²)"] > 1e-4]
        df_interface_all = pd.DataFrame(df_interface_all_data)

        if not df_pairs.empty:
            df_pairs.index = df_pairs.index + 1
        if not df_atoms.empty:
            df_atoms.index = df_atoms.index + 1
        if not df_interface_all.empty:
            df_interface_all.index = df_interface_all.index + 1

        # Populate shared results map
        analysis_results = {
            "df_pairs": df_pairs,
            "df_atoms": df_atoms,
            "df_interface_all": df_interface_all,
            "total_excluded_sasa": total_excluded_sasa,
            "interface_classification": interface_classification,
            "all_residue_sasa_records": all_residue_sasa_records,
            "contact_links_visualization": contact_links_visualization,
            "total_unique_res": len(df_interface_all),
            "chain_a": chain_a,
            "chain_b": chain_b,
            "selected_cutoff": selected_cutoff,
            "pdb_raw_data": pdb_raw_data,
            "pdb_id": pdb_id
        }

# Main Tab 1: Primary Analysis Section (3D Viewer)
with main_tab_analysis:
    if not analysis_results:
        st.info("⬜ Please load a structure from the sidebar & select valid parameters to begin.")
    else:
        df_pairs = analysis_results["df_pairs"]
        df_atoms = analysis_results["df_atoms"]
        df_interface_all = analysis_results["df_interface_all"]
        total_excluded_sasa = analysis_results["total_excluded_sasa"]
        interface_classification = analysis_results["interface_classification"]
        all_residue_sasa_records = analysis_results["all_residue_sasa_records"]
        contact_links_visualization = analysis_results["contact_links_visualization"]
        total_unique_res = analysis_results["total_unique_res"]
        chain_a = analysis_results["chain_a"]
        chain_b = analysis_results["chain_b"]
        selected_cutoff = analysis_results["selected_cutoff"]
        pdb_raw_data = analysis_results["pdb_raw_data"]
        pdb_id = analysis_results["pdb_id"]

        col_visualizer, col_filters = st.columns([3, 1])

        with col_filters:
            st.markdown('<div class="sidebar-header-highlight">3. 3D Filter Display</div>', unsafe_allow_html=True)
            enable_labels = st.checkbox("Enable Residue Labels", value=True)
            show_molecular_surface = st.checkbox("Transparent molecular surface", value=False)

            if show_molecular_surface:
                surface_opacity = st.slider("Surface Opacity", min_value=0.1, max_value=1.0, value=0.4, step=0.1)
            else:
                surface_opacity = 0.4

            if contact_links_visualization:
                available_pairs = [link["pair_label"] for link in contact_links_visualization]
                selected_visual_pairs = st.multiselect(
                    "Select contact residue pairs to display (leave empty to show all):",
                    options=available_pairs,
                    default=[]
                )
                if selected_visual_pairs:
                    filtered_links = [link for link in contact_links_visualization if link["pair_label"] in selected_visual_pairs]
                else:
                    filtered_links = contact_links_visualization
            else:
                st.info("No contacts available to filter.")
                filtered_links = []

            bg_color_option = st.selectbox("Viewer Background Color:", options=["black", "white", "gray"], format_func=lambda x: x.capitalize(), index=0)

            color_choices = {
                "Blue": "#3B82F6",
                "Red": "#EF4444",
                "Green": "#10B981",
                "Purple": "#8B5CF6",
                "Orange": "#F59E0B",
                "Yellow": "#FBBF24",
                "Pink": "#EC4899",
                "Cyan": "#06B6D4"
            }
            chain_a_color_name = st.selectbox(f"Color for Chain 1 ({chain_a}):", options=list(color_choices.keys()), index=0)
            chain_b_color_name = st.selectbox(f"Color for Chain 2 ({chain_b}):", options=list(color_choices.keys()), index=1)

            chain_a_hex = color_choices[chain_a_color_name]
            chain_b_hex = color_choices[chain_b_color_name]

        with col_visualizer:
            st.subheader("⚡ 3D Molecular Interface Visualization")
            
            # Construct the py3Dmol component
            py3dmol_html = f"""
            <div style="height: 520px; width: 100%; border: 1px solid #E5E7EB; border-radius: 8px; overflow: hidden;">
                <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
                <div id="py3dmol_viewer" style="width: 100%; height: 100%; position: relative;"></div>
                <script>
                    document.addEventListener("DOMContentLoaded", function() {{
                        var element = document.getElementById("py3dmol_viewer");
                        var config = {{ backgroundColor: "{bg_color_option}" }};
                        var viewer = $3Dmol.createViewer(element, config);
                        
                        var pdbData = `{pdb_raw_data}`;
                        viewer.addModel(pdbData, "pdb");
                        
                        // Styles Chains
                        viewer.setStyle({{chain: "{chain_a}"}}, {{cartoon: {{color: "{chain_a_hex}"}}}});
                        viewer.setStyle({{chain: "{chain_b}"}}, {{cartoon: {{color: "{chain_b_hex}"}}}});
            """

            # Highlight contact links with solid cylinders
            for link in filtered_links:
                spec_a = link["spec_a"]
                spec_b = link["spec_b"]
                c_a = link["coord_a"]
                c_b = link["coord_b"]

                # Highlight side-chain atoms of interacting residues
                py3dmol_html += f"""
                        viewer.setStyle({{chain: "{spec_a['chain']}", resi: {spec_a['resi']}}}, {{stick: {{radius: 0.15, colorscheme: 'carbon'}}}});
                        viewer.setStyle({{chain: "{spec_b['chain']}", resi: {spec_b['resi']}}}, {{stick: {{radius: 0.15, colorscheme: 'carbon'}}}});
                        
                        viewer.addCylinder({{
                            start: {{x: {c_a[0]}, y: {c_a[1]}, z: {c_a[2]}}},
                            end: {{x: {c_b[0]}, y: {c_b[1]}, z: {c_b[2]}}},
                            radius: 0.08,
                            color: "#F59E0B",
                            fromCap: true,
                            toCap: true
                        }});
                """

                if enable_labels:
                    py3dmol_html += f"""
                        viewer.addLabel("{link['label_a'].strip()}", {{fontSize: 10, backgroundOpacity: 0.7, backgroundColor: "#1F2937", fontColor: "#FFFFFF", position: {{x: {c_a[0]}, y: {c_a[1]}, z: {c_a[2]}}}});
                        viewer.addLabel("{link['label_b'].strip()}", {{fontSize: 10, backgroundOpacity: 0.7, backgroundColor: "#1F2937", fontColor: "#FFFFFF", position: {{x: {c_b[0]}, y: {c_b[1]}, z: {c_b[2]}}}});
                    """

            if show_molecular_surface:
                py3dmol_html += f"""
                        viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: {surface_opacity}, color: "#D1D5DB"}}, {{chain: "{chain_a}"}});
                        viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: {surface_opacity}, color: "#9CA3AF"}}, {{chain: "{chain_b}"}});
                """

            py3dmol_html += """
                        viewer.zoomTo();
                        viewer.render();
                    });
                </script>
            </div>
            """
            components.html(py3dmol_html, height=540)

        st.markdown("--- ")
        st.subheader("📊 Structural Metrics & Surface Area Calculations")

        # Standard metric cards
        c_card1, c_card2, c_card3, c_card4 = st.columns(4)
        c_card1.metric("Atom-Atom Contacts", len(df_atoms))
        c_card2.metric("Residue-Residue Pairs", len(df_pairs))
        c_card3.metric("Unique Interface Residues", total_unique_res)
        
        # Excluded SASA serves as the physical Buried Surface Area (BSA)
        c_card4.metric("Buried Surface Area (BSA)", f"{round(total_excluded_sasa, 2)} Å²")

        st.info(f"🔬 **Interface Structural Classification**: {interface_classification}")

        # Sub-expanders detailing calculated records
        with st.expander("i) Unique Interface Residues & SASA Properties", expanded=True):
            if not df_interface_all.empty:
                st.dataframe(df_interface_all, use_container_width=True)
            else:
                st.warning("No solvent-accessible surface modifications detected at current thresholds.")

        with st.expander("ii) Residue-Residue Interactive Pairs", expanded=False):
            if not df_pairs.empty:
                st.dataframe(df_pairs, use_container_width=True)
            else:
                st.warning("No residue-residue contacts registered.")

        with st.expander("iii) Detailed Atom-Atom Contacts Matrix", expanded=False):
            if not df_atoms.empty:
                st.dataframe(df_atoms, use_container_width=True)
            else:
                st.warning("No atom-atom contacts registered within threshold limitations.")

        # Download Reports Section
        st.markdown("### 📥 Export Structural Reports")
        col_export_pdf, col_export_csv = st.columns(2)

        with col_export_csv:
            csv_buffer = io.StringIO()
            df_interface_all.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Interface Residues as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{pdb_id}_interface_residues.csv",
                mime="text/csv"
            )

        with col_export_pdf:
            try:
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                story = []
                styles = getSampleStyleSheet()
                
                # Custom ReportLab document styles
                title_style = ParagraphStyle(
                    'ReportTitle',
                    parent=styles['Heading1'],
                    fontSize=20,
                    textColor=colors.HexColor('#1E3A8A'),
                    spaceAfter=15
                )
                body_style = ParagraphStyle(
                    'ReportBody',
                    parent=styles['Normal'],
                    fontSize=10,
                    spaceAfter=10
                )
                
                story.append(Paragraph(f"Protein Contact Residue Server Analysis: {pdb_id.upper()}", title_style))
                story.append(Paragraph(f"Calculated Parameters: Chain 1: {chain_a} | Chain 2: {chain_b} | Cut-off: {selected_cutoff} Å", body_style))
                story.append(Paragraph(f"<b>Interface Size (Buried Surface Area):</b> {round(total_excluded_sasa, 2)} Å²", body_style))
                story.append(Paragraph(f"<b>Classification:</b> {interface_classification}", body_style))
                story.append(Spacer(1, 15))

                # Residue-Residue table conversion
                if not df_pairs.empty:
                    table_data = [['Residue A', 'Residue B', 'Min Dist (Å)', 'Closest Atoms']]
                    for idx, row in df_pairs.head(25).iterrows():
                        table_data.append([
                            f"{row['Residue A Name']}{row['Residue A ID']}",
                            f"{row['Residue B Name']}{row['Residue B ID']}",
                            str(row['Minimum Distance (Å)']),
                            str(row['Closest Atoms (A-B)'])
                        ])
                    pdf_table = Table(table_data)
                    pdf_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 10),
                        ('BOTTOMPADDING', (0,0), (-1,0), 6),
                        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#E5E7EB')),
                        ('FONTSIZE', (0,1), (-1,-1), 8),
                    ]))
                    story.append(Paragraph("<b>Pairwise Residue Interface Interactions (Top 25)</b>", styles['Heading2']))
                    story.append(pdf_table)

                doc.build(story)
                pdf_bytes = pdf_buffer.getvalue()

                st.download_button(
                    label="Download Structural Summary as PDF",
                    data=pdf_bytes,
                    file_name=f"{pdb_id}_interface_report.pdf",
                    mime="application/pdf"
                )
            except Exception as pdf_err:
                st.error(f"Unable to format PDF binary stream: {pdf_err}")

# Main Tab 2: Custom 2D Schematic Interface Diagram with Dynamic Group-Based Coloring
with main_tab_2d_viewer:
    st.subheader("▧ Inter-chain 2D Interface Diagram")
    st.markdown("""<div class="justified-para">
    This tab renders a publication-ready schematic diagram representing individual inter-residue contacts as linkages. 
    Each residue node is dynamically colored based on its chemical <b>Group Distribution</b> (Hydrophobic, Polar, 
    Charged, Aromatic, or standard canonical amino acid types) to help you quickly identify stabilizing networks.
    </div>""", unsafe_allow_html=True)

    if not analysis_results:
        st.info("⬜ Please load a structure from the sidebar & select valid parameters to begin.")
    else:
        df_pairs = analysis_results["df_pairs"]
        chain_a = analysis_results["chain_a"]
        chain_b = analysis_results["chain_b"]

        if df_pairs.empty:
            st.warning("No contacts detected to plot.")
        else:
            # Isolate contact pairings
            unique_res_a = sorted(list(set(df_pairs["Residue A Name"] + df_pairs["Residue A ID"].astype(str))))
            unique_res_b = sorted(list(set(df_pairs["Residue B Name"] + df_pairs["Residue B ID"].astype(str))))

            # Map specific residues to positions
            pos_a = {res: (1.0, i) for i, res in enumerate(unique_res_a)}
            pos_b = {res: (3.0, i) for i, res in enumerate(unique_res_b)}

            # Function to determine color based on operational chemical classifications
            def get_residue_color(residue_str):
                res_name_3 = residue_str[:3].upper()
                if res_name_3 in amino_acid_groups['Hydrophobic'] and res_name_3 not in amino_acid_groups['Aromatic']:
                    return '#3B82F6'  # Hydrophobic -> Blue
                elif res_name_3 in amino_acid_groups['Polar']:
                    return '#10B981'  # Polar -> Green
                elif res_name_3 in amino_acid_groups['Positively charged']:
                    return '#8B5CF6'  # Positively charged -> Purple
                elif res_name_3 in amino_acid_groups['Negatively charged']:
                    return '#EF4444'  # Negatively charged -> Red
                elif res_name_3 in amino_acid_groups['Aromatic']:
                    return '#F59E0B'  # Aromatic -> Orange
                else:
                    return '#4B5563'  # Canonical/Other (e.g. GLY) -> Dark Gray

            # Plot configuration with adjusted dynamic canvas height
            fig_height = max(5, max(len(unique_res_a), len(unique_res_b)) * 0.4)
            fig, ax = plt.subplots(figsize=(10, fig_height))

            # Plot interface contact linkages
            for _, row in df_pairs.iterrows():
                node_a_label = f"{row['Residue A Name']}{row['Residue A ID']}"
                node_b_label = f"{row['Residue B Name']}{row['Residue B ID']}"
                x_coords = [pos_a[node_a_label][0], pos_b[node_b_label][0]]
                y_coords = [pos_a[node_a_label][1], pos_b[node_b_label][1]]
                ax.plot(x_coords, y_coords, color="#9CA3AF", linestyle="-", linewidth=1.2, alpha=0.7, zorder=1)

            # Plot Chain A nodes
            for res, (x, y) in pos_a.items():
                node_color = get_residue_color(res)
                ax.scatter(x, y, s=280, color=node_color, edgecolors="black", zorder=2)
                ax.text(x - 0.12, y, res, ha="right", va="center", fontweight="bold", fontsize=9, color="#1F2937")

            # Plot Chain B nodes
            for res, (x, y) in pos_b.items():
                node_color = get_residue_color(res)
                ax.scatter(x, y, s=280, color=node_color, edgecolors="black", zorder=2)
                ax.text(x + 0.12, y, res, ha="left", va="center", fontweight="bold", fontsize=9, color="#1F2937")

            # Axes formatting
            ax.set_xlim(0.2, 3.8)
            ax.set_ylim(-1, max(len(unique_res_a), len(unique_res_b)))
            ax.axis("off")

            # Chain headers
            ax.text(1.0, max(len(unique_res_a), len(unique_res_b)) - 0.2, f"Chain 1: {chain_a}", 
                    ha="center", va="bottom", fontsize=13, fontweight="bold", color="#1E3A8A", bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#DBEAFE"))
            ax.text(3.0, max(len(unique_res_a), len(unique_res_b)) - 0.2, f"Chain 2: {chain_b}", 
                    ha="center", va="bottom", fontsize=13, fontweight="bold", color="#1E3A8A", bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#DBEAFE"))

            st.pyplot(fig)

            # Color-coded Legend explaining classification
            st.markdown("""
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 16px; margin-top: 20px; padding: 10px; background-color: #F9FAFB; border-radius: 8px; border: 1px solid #F3F4F6;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 14px; background-color: #3B82F6; border: 1px solid #1D4ED8; border-radius: 50%;"></span>
                    <span style="font-size: 0.85rem; font-weight: 600; color: #374151;">Hydrophobic</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 14px; background-color: #10B981; border: 1px solid #047857; border-radius: 50%;"></span>
                    <span style="font-size: 0.85rem; font-weight: 600; color: #374151;">Polar</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 14px; background-color: #8B5CF6; border: 1px solid #6D28D9; border-radius: 50%;"></span>
                    <span style="font-size: 0.85rem; font-weight: 600; color: #374151;">Positively Charged</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 14px; background-color: #EF4444; border: 1px solid #B91C1C; border-radius: 50%;"></span>
                    <span style="font-size: 0.85rem; font-weight: 600; color: #374151;">Negatively Charged</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 14px; background-color: #F59E0B; border: 1px solid #B45309; border-radius: 50%;"></span>
                    <span style="font-size: 0.85rem; font-weight: 600; color: #374151;">Aromatic</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 14px; background-color: #4B5563; border: 1px solid #1F2937; border-radius: 50%;"></span>
                    <span style="font-size: 0.85rem; font-weight: 600; color: #374151;">Other / Canonical (Glycine, etc.)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Detailed chemical composition tables within the 2D View Tab
            st.markdown("### 📊 Interface Composition Statistics")
            unique_interface_all_res = list(set(df_interface_all["Residue"]))
            df_aa_stats, df_groups, aa_counts = get_amino_acid_statistics(unique_interface_all_res)

            c_tbl1, c_tbl2 = st.columns(2)
            with c_tbl1:
                st.markdown("**i) Group Distributions Across Interface**")
                st.dataframe(df_groups, use_container_width=True)
            with c_tbl2:
                st.markdown("**ii) Single Amino Acid Frequencies**")
                st.dataframe(df_aa_stats, use_container_width=True)
