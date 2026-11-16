

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

# Display attached user image if available
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
    The **3D Filter Display** panel allows you to customize and focus on specific structural elements inside the 3D canvas:
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

# Initialize analysis containers
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
                        "Residue": res_name_full,
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
        st.info("◽ Please load a structure from the sidebar & select valid parameters to begin.")
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
            surface_opacity = st.slider("Surface Opacity", 0.1, 1.0, 0.4, 0.1) if show_molecular_surface else 0.4

            if contact_links_visualization:
                available_pairs = [link["pair_label"] for link in contact_links_visualization]
                selected_visual_pairs = st.multiselect(
                    "Select contact residue pairs:", options=available_pairs, default=[]
                )
                filtered_links = [link for link in contact_links_visualization if link["pair_label"] in selected_visual_pairs] if selected_visual_pairs else contact_links_visualization
            else:
                st.info("No contacts available.")
                filtered_links = []

            bg_color_option = st.selectbox("Background:", ["black", "white", "gray"], index=0)
            color_choices = {"Blue": "#3B82F6", "Red": "#EF4444", "Green": "#10B981", "Purple": "#8B5CF6", "Orange": "#F59E0B"}
            chain_a_color = color_choices[st.selectbox(f"Chain {chain_a} Color:", list(color_choices.keys()), index=0)]
            chain_b_color = color_choices[st.selectbox(f"Chain {chain_b} Color:", list(color_choices.keys()), index=1)]

        with col_visualizer:
            st.subheader(f"3D Molecular View: Chain {chain_a} and Chain {chain_b}")
            if pdb_raw_data:
                try:
                    escaped_pdb = pdb_raw_data.replace("\n", "\\n").replace("\"", "\\\"").replace("'", "\\\'")
                    cylinder_lines_js = ""
                    labels_js = ""
                    sidechains_js = ""
                    vdw_surface_js = ""

                    for link in filtered_links:
                        ca, cb = link["coord_a"], link["coord_b"]
                        cylinder_lines_js += f"""
                        viewer.addCylinder({{
                            start: {{x: {ca[0]}, y: {ca[1]}, z: {ca[2]}}},
                            end: {{x: {cb[0]}, y: {cb[1]}, z: {cb[2]}}},
                            radius: 0.15, fromCap: 1, toCap: 1, color: 'yellow', dashed: true
                        }});
                        """
                        if enable_labels:
                            labels_js += f"""
                            viewer.addLabel('{link["label_a"]}', {{fontColor:'black', backgroundColor:'#DCE7F3', fontSize:10, backgroundOpacity:0.8, borderThickness:1.0, borderColor:'#3B82F6'}}, {{chain: '{link["spec_a"]["chain"]}', resi: {link["spec_a"]["resi"]}}});
                            viewer.addLabel('{link["label_b"]}', {{fontColor:'black', backgroundColor:'#FEE2E2', fontSize:10, backgroundOpacity:0.8, borderThickness:1.0, borderColor:'#EF4444'}}, {{chain: '{link["spec_b"]["chain"]}', resi: {link["spec_b"]["resi"]}}});
                            """
                        sidechains_js += f"""
                        viewer.setStyle({{chain: '{link["spec_a"]["chain"]}', resi: {link["spec_a"]["resi"]}}}, {{stick: {{colorspec: 'cyan'}}}});
                        viewer.setStyle({{chain: '{link["spec_b"]["chain"]}', resi: {link["spec_b"]["resi"]}}}, {{stick: {{colorspec: 'orange'}}}});
                        """
                        if show_molecular_surface:
                            vdw_surface_js += f"""
                            viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: {surface_opacity}, color: '{chain_a_color}'}}, {{chain: '{link["spec_a"]["chain"]}', resi: {link["spec_a"]["resi"]}}});
                            viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: {surface_opacity}, color: '{chain_b_color}'}}, {{chain: '{link["spec_b"]["chain"]}', resi: {link["spec_b"]["resi"]}}});
                            """

                    py3dmol_html = f"""
                    <div id="3dmol_canvas" style="width: 100%; height: 500px; position: relative"></div>
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/1.7.0/3Dmol-min.js"></script>
                    <script>
                        var element = document.getElementById('3dmol_canvas');
                        var viewer = $3Dmol.createViewer(element, {{ backgroundColor: '{bg_color_option}' }});
                        viewer.addModel('{escaped_pdb}', 'pdb');
                        viewer.setStyle({{chain: '{chain_a}'}}, {{cartoon: {{color: '{chain_a_color}'}}}});
                        viewer.setStyle({{chain: '{chain_b}'}}, {{cartoon: {{color: '{chain_b_color}'}}}});
                        {cylinder_lines_js}
                        {sidechains_js}
                        {labels_js}
                        {vdw_surface_js}
                        viewer.zoomTo({{chain: ['{chain_a}', '{chain_b}']}});
                        viewer.render();
                    </script>
                    """
                    components.html(py3dmol_html, height=520)
                except Exception as e:
                    st.error(f"Viewer error: {e}")

        # Analysis Metrics and Tables
        st.subheader(f"Inter-chain Contacts: Chain {chain_a} and Chain {chain_b} at {selected_cutoff} Å")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Unique Interface Residues", f" {total_unique_res}")
        col_m2.metric("Residue–Residue Pairs", len(df_pairs))
        col_m3.metric("Buried Surface Area (BSA)", f"{total_excluded_sasa:.1f} Å²")
        col_m4.metric("Interface Classification", interface_classification.split(" (")[0])

        if not df_pairs.empty:
            tab1, tab2, tab3 = st.tabs(["Unique Residues SASA", "Residue-Residue Pairs", "Detailed Atom Contacts"])
            with tab1:
                st.dataframe(df_interface_all, use_container_width=True)
                df_aa_stats, df_groups, aa_counts = get_amino_acid_statistics(df_interface_all["Residue"].tolist())
                st.dataframe(df_aa_stats, use_container_width=True)
            with tab2:
                st.dataframe(df_pairs, use_container_width=True)
            with tab3:
                st.dataframe(df_atoms, use_container_width=True)

# Main Tab 2: 2D Interactive Schematic Diagram Viewer
with main_tab_2d_viewer:
    if not analysis_results:
        st.info("◽ Please load a structure from the sidebar first.")
    else:
        df_pairs = analysis_results["df_pairs"]
        if df_pairs.empty:
            st.warning("No contact residue pairs found.")
        else:
            unique_nodes_a = sorted(list(set(df_pairs.apply(lambda r: f"{r['Residue A Name']}{r['Residue A ID']}", axis=1))))
            unique_nodes_b = sorted(list(set(df_pairs.apply(lambda r: f"{r['Residue B Name']}{r['Residue B ID']}", axis=1))))
            connections = []
            for _, row in df_pairs.iterrows():
                res_a = f"{row['Residue A Name']}{row['Residue A ID']}"
                res_b = f"{row['Residue B Name']}{row['Residue B ID']}"
                connections.append({"from": res_a, "to": res_b, "dist": row["Minimum Distance (Å)"]})

            show_distance_labels = st.checkbox("Display distance labels on schematic connections", value=True)
            svg_width = 800
            node_height = 35
            node_width = 110
            spacing = 15
            total_height = max(len(unique_nodes_a), len(unique_nodes_b)) * (node_height + spacing) + 80

            import json
            connections_json = json.dumps(connections)
            nodes_a_json = json.dumps(unique_nodes_a)
            nodes_b_json = json.dumps(unique_nodes_b)
            show_distance_val = "true" if show_distance_labels else "false"

            schematic_html = f"""
            <div style="width:100%; overflow-x:auto;">
                <svg id="schematic_2d" width="{svg_width}" height="{total_height}" style="background:#FFFFFF; border: 1px solid #E5E7EB; border-radius:12px; margin: 10px auto; display:block;">
                    <defs>
                        <linearGradient id="chainAGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#3B82F6" />
                            <stop offset="100%" stop-color="#1D4ED8" />
                        </linearGradient>
                        <linearGradient id="chainBGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#EF4444" />
                            <stop offset="100%" stop-color="#B91C1C" />
                        </linearGradient>
                    </defs>
                    <text x="150" y="30" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="#1E3A8A">Chain {chain_a} Residues</text>
                    <text x="650" y="30" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="#B91C1C">Chain {chain_b} Residues</text>
                    <g id="linkages_layer"></g>
                    <g id="nodes_layer"></g>
                </svg>
            </div>
            <script>
                const nodesA = {nodes_a_json};
                const nodesB = {nodes_b_json};
                const links = {connections_json};
                const showDist = {show_distance_val};
                const nodeW = {node_width}, nodeH = {node_height}, spacing = {spacing};
                const xA = 100, xB = 590, yStart = 60;
                const posA = {{}}, posB = {{}};

                const nl = document.getElementById('nodes_layer');
                const ll = document.getElementById('linkages_layer');

                nodesA.forEach((n, i) => {{
                    const y = yStart + i * (nodeH + spacing);
                    posA[n] = {{ x: xA + nodeW, y: y + nodeH / 2 }};
                    nl.innerHTML += `<rect x="${{xA}}" y="${{y}}" width="${{nodeW}}" height="${{nodeH}}" rx="6" fill="url(#chainAGrad)" />
                                     <text x="${{xA+nodeW/2}}" y="${{y+nodeH/2+4}}" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="11" fill="#FFFFFF">${{n}}</text>`;
                }});

                nodesB.forEach((n, i) => {{
                    const y = yStart + i * (nodeH + spacing);
                    posB[n] = {{ x: xB, y: y + nodeH / 2 }};
                    nl.innerHTML += `<rect x="${{xB}}" y="${{y}}" width="${{nodeW}}" height="${{nodeH}}" rx="6" fill="url(#chainBGrad)" />
                                     <text x="${{xB+nodeW/2}}" y="${{y+nodeH/2+4}}" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="11" fill="#FFFFFF">${{n}}</text>`;
                }});

                links.forEach(l => {{
                    const pA = posA[l.from], pB = posB[l.to];
                    if (pA && pB) {{
                        const mx = (pA.x + pB.x)/2, my = (pA.y + pB.y)/2;
                        const formattedD = parseFloat(l.dist).toFixed(2);
                        let lbl = showDist ? `<rect x="${{mx - 22}}" y="${{my - 8}}" width="44" height="16" rx="4" fill="#F3F4F6" stroke="#D1D5DB" />
                                              <text x="${{mx}}" y="${{my + 4}}" text-anchor="middle" font-family="sans-serif" font-size="9" font-weight="bold" fill="#374151">${{formattedD}}Å</text>` : '';
                        ll.innerHTML += `<line x1="${{pA.x}}" y1="${{pA.y}}" x2="${{pB.x}}" y2="${{pB.y}}" stroke="#9CA3AF" stroke-width="2" stroke-dasharray="4,4" />${{lbl}}`;
                    }}
                }});
            </script>
            """
            components.html(schematic_html, height=total_height + 50)
