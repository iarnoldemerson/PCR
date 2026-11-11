

import streamlit as st
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, PDBList
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
st.markdown("<div class=\"main-title\">⚕️ Protein Contact Residue Server</div>", unsafe_allow_html=True)
st.write("Determine contact residues between two protein chains using customizable distance thresholds.")

# Create Main Tabs at the top of the page
main_tab_analysis, main_tab_help = st.tabs([
    "🔬 Interface Analysis & 3D Viewer",
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
st.sidebar.markdown("<div class=\"sidebar-header-highlight\">1. Load PDB Structure</div>", unsafe_allow_html=True)
input_method = st.sidebar.radio("Choose Source:", ("Fetch from PDB (RCSB)", "Upload PDB File"))

parser = PDBParser(QUIET=True)
structure = None
pdb_raw_data = None
pdb_id = ""

# Initialize Streamlit session state for selected structure details
if "pdb_id_to_load" not in st.session_state:
    st.session_state.pdb_id_to_load = ""

if input_method == "Fetch from PDB (RCSB)":
    # Provide demo load button
    if st.sidebar.button("⚡ Load Demo Structure (6Q9O)"):
        st.session_state.pdb_id_to_load = "6Q9O"
        st.rerun()

    pdb_input = st.sidebar.text_input(
        "Enter 4-letter PDB ID (e.g., 6Q9O):",
        value=st.session_state.pdb_id_to_load,
        key="pdb_text_field"
    ).strip()

    # Keep session state in sync with text field input manually
    if pdb_input != st.session_state.pdb_id_to_load:
        st.session_state.pdb_id_to_load = pdb_input

    if pdb_input:
        pdb_id = pdb_input
        try:
            pdbl = PDBList()
            pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="pdb")
            with open(pdb_filename, "r") as f:
                pdb_raw_data = f.read()
            structure = parser.get_structure(pdb_id, pdb_filename)
            st.sidebar.success(f"Successfully loaded structure for PDB ID: {pdb_id.upper()}")
        except Exception as e:
            st.sidebar.error(f"Error fetching PDB ID {pdb_id}: {e}")

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

# Main Tab 2: Help Guide (Rendered at top of the page)
with main_tab_help:
    st.markdown("### 📘 Structural Interface Analysis Guide")
    st.write("Welcome to the reference guide and operating manual for this application. Below are step-by-step instructions for executing your contact calculations and interacting with your results.")

    st.markdown("#### 🚀 Step-by-Step Instructions")
    st.markdown("""
    1. **Load Your Protein Structure**:
       * Look at **Section 1** in the left sidebar.
       * Click **⚡ Load Demo Structure (6Q9O)** to try the application immediately, or enter any valid 4-character PDB ID into the text input, or choose **Upload PDB File** to provide a local `.pdb` file directly from your computer.

    2. **Configure Your Chains and Threshold**:
       * Once the structure is loaded, **Section 2** in the sidebar activates.
       * Pick **Chain 1** and **Chain 2** from the dropdowns.
       * Set your desired **Distance Cut-off** using the dropdown options ($4.0\\text{ Å}$, $5.0\\text{ Å}$, or $6.0\\text{ Å}$).

    3. **Interact and Filter the 3D View**:
       * The **3D Molecular View** displays cartoons of both chains, linked by yellow dashed lines at the contact coordinates.
       * Use the right-hand **3D Filter Display** panel to turn labels on/off, change the viewer background, and customize cartoon chain colors.
       * To focus on a subset of residues, select them in the **\"Select contact residue pairs to display\"** multi-select box.
       * Click **Reset 3D View** to restore the default camera, or **Save 3D View as PNG Image** to save a snapshot.

    4. **Examine Tabular Analysis & Download CSVs**:
       * Click through the tabs below the viewer to study the results:
         * **🧬 Unique Interface Residues**: Lists unique interface residues and chemical grouping charts.
         * **🔗 Residue-Residue Pairs**: Lists closest atom pairings with contact distances.
         * **🔍 Detailed Atom-Atom Contacts**: Explores contacts on an individual atomic coordinate level.
       * Click the **\"Download CSV\"** buttons inside each tab to download that specific spreadsheet.

    5. **Export Your Report**:
       * Scroll to the bottom of the page to find the **Export Report as PDF** section.
       * Click **Download PDF Report** to generate a formatted document complete with your metrics, chemical charts, distance histogram, and a contact table.
    """)

    st.markdown("#### 🧬 Understanding the Metrics")
    st.markdown("""
    * **Unique Interface Residues**:
      * *What it is*: The absolute count of distinct residues participating in contacts across both chains.
      * *Why it matters*: If Residue `ALA12` on Chain A is close enough to interact with both `VAL40` and `LEU42` on Chain B, `ALA12` is only counted **once** as a unique interface residue. This metric is ideal for estimating the overall surface area of the active interface region.
    * **Residue-Residue Pairs**:
      * *What it is*: Unique combinations of contacting residue pairs (e.g., `ALA12` - `VAL40`).
      * *Why it matters*: Unlike the atom count, this is a sequence-level metric. If multiple atoms in `ALA12` interact with multiple atoms in `VAL40`, it still yields exactly one residue pair contact.
    * **Atom-Atom Contacts**:
      * *What it is*: The raw physical contacts occurring at the atomic level (e.g., side-chain carrots or backbone oxygens/nitrogens within the cut-off boundary).
      * *Why it matters*: This metric describes the physical intensity or packing density of the interface. A high count of atomic contacts between a residue pair usually implies tightly nested packing or complex hydrogen bonding networks.
    """)

    st.markdown("#### 📏 Distance Cut-off Reference")
    st.markdown("""
    * **4.0 Å (Close Contacts)**: Ideal for highlighting specific high-affinity interactions such as hydrogen bonds, salt bridges, and closely-packed hydrophobic interactions.
    * **5.0 Å (Standard Contact Definition)**: The classical structural biology default. Captures both specific bonds and secondary van der Waals interactions across the binding interface.
    * **6.0 Å (Broader Interfaces)**: Useful for identifying flanking, solvent-accessible residues that define the broader hydration shell or structural pocket of the interface.
    """)

# Main Tab 1: Primary Analysis Section
with main_tab_analysis:
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
                                "Closest Atoms (A-B)": f"{(closest_atoms[0] or '')}-{(closest_atoms[1] or '')}"
                            }
                            if visual_atom_pair is not None:
                                contact_links_visualization.append({
                                    "pair_label": f"{res_a.get_resname()}{res_a.id[1]} - {res_b.get_resname()}{res_b.id[1]}",
                                    "coord_a": visual_atom_pair[0],
                                    "coord_b": visual_atom_pair[1],
                                    "label_a": f"{res_a.get_resname()}{res_a.id[1]}",
                                    "label_b": f"{res_b.get_resname()}{res_b.id[1]}",
                                    "spec_a": { "chain": chain_a, "resi": res_a.id[1] },
                                    "spec_b": { "chain": chain_b, "resi": res_b.id[1] }
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

                    # Customize the visualizer background color
                    bg_color_option = st.selectbox("Viewer Background Color:", options=["black", "white", "gray"], format_func=lambda x: x.capitalize(), index=0)

                    # Customize chain cartoon colors
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
                    chain_a_color_name = st.selectbox(f"Chain {chain_a} Color:", options=list(color_choices.keys()), index=0)
                    chain_b_color_name = st.selectbox(f"Chain {chain_b} Color:", options=list(color_choices.keys()), index=1)
                    chain_a_color = color_choices[chain_a_color_name]
                    chain_b_color = color_choices[chain_b_color_name]

                with col_visualizer:
                    st.subheader(f"🗾 3D Molecular View: Chain {chain_a} and Chain {chain_b}")
                    st.write("Interactive 3D representation showing contacting residues linked together.")

                    if pdb_raw_data:
                        try:
                            # Format molecular instructions using direct py3Dmol/javascript scripting wrapper
                            escaped_pdb = pdb_raw_data.replace("\n", "\\n").replace("\"", "\\\"").replace("'", "\\\'")

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
                                # Labels rendering restored to original bordered design
                                if enable_labels:
                                    labels_js += f"""
                                    viewer.addLabel('{link["label_a"]}', {{fontColor:'black', backgroundColor:'#DCE7F3', fontSize:10, backgroundOpacity:0.8, borderThickness:1.0, borderColor:'#3B82F6'}}, {{chain: '{link["spec_a"]["chain"]}', resi: {link["spec_a"]["resi"]}}});
                                    viewer.addLabel('{link["label_b"]}', {{fontColor:'black', backgroundColor:'#FEE2E2', fontSize:10, backgroundOpacity:0.8, borderThickness:1.0, borderColor:'#EF4444'}}, {{chain: '{link["spec_b"]["chain"]}', resi: {link["spec_b"]["resi"]}}});
                                    """
                                # Sidechains (Stick styling)
                                sidechains_js += f"""
                                viewer.setStyle({{chain: '{link["spec_a"]["chain"]}', resi: {link["spec_a"]["resi"]}}}, {{stick: {{colorspec: 'cyan'}}}});
                                viewer.setStyle({{chain: '{link["spec_b"]["chain"]}', resi: {link["spec_b"]["resi"]}}}, {{stick: {{colorspec: 'orange'}}}});
                                """

                            # Clean iframe component view with 'Capture Snapshot' buttons removed
                            py3dmol_html = f"""
                            <div id=\"3dmol_canvas\" style=\"width: 100%; height: 500px; position: relative;\"></div>
                            <div style=\"margin-top: 10px; margin-bottom: 20px; display: flex; gap: 10px;\">
                                <button id=\"reset_view_btn\" style=\"
                                    background-color: #6B7280;
                                    color: white;
                                    border: none;
                                    padding: 8px 16px;
                                    border-radius: 6px;
                                    font-weight: 600;
                                    cursor: pointer;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                \">🔄 Reset 3D View</button>
                                <button id=\"download_png_btn\" style=\"
                                    background-color: #2563EB;
                                    color: white;
                                    border: none;
                                    padding: 8px 16px;
                                    border-radius: 6px;
                                    font-weight: 600;
                                    cursor: pointer;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                \">📸 Save 3D View as PNG Image</button>
                            </div>
                            <script src=\"https://cdnjs.cloudflare.com/ajax/libs/3Dmol/1.7.0/3Dmol-min.js\"></script>
                            <script>
                                var element = document.getElementById('3dmol_canvas');
                                var config = {{ backgroundColor: '{bg_color_option}' }};
                                var viewer = $3Dmol.createViewer(element, config);
                                viewer.addModel('{escaped_pdb}', 'pdb');

                                // Clear styling of all molecules/chains
                                viewer.setStyle({{}}, {{}});

                                // Display ONLY the selected interacting chains with customized colors
                                viewer.setStyle({{chain: '{chain_a}'}}, {{cartoon: {{color: '{chain_a_color}'}}}});
                                viewer.setStyle({{chain: '{chain_b}'}}, {{cartoon: {{color: '{chain_b_color}'}}}});

                                {cylinder_lines_js}
                                {sidechains_js}
                                {labels_js}
                                viewer.zoomTo({{chain: ['{chain_a}', '{chain_b}']}});
                                viewer.render();

                                // Reset camera orientation and zoom
                                document.getElementById('reset_view_btn').addEventListener('click', function() {{
                                    viewer.zoomTo({{chain: ['{chain_a}', '{chain_b}']}});
                                    viewer.render();
                                }});

                                // Interactive canvas to image snapshot exporter (PNG)
                                document.getElementById('download_png_btn').addEventListener('click', function() {{
                                    try {{
                                        var canvas = document.getElementsByTagName('canvas')[0];
                                        if (!canvas) throw new Error(\"Canvas element not found.\");
                                        var dataUrl = canvas.toDataURL('image/png');
                                        var link = document.createElement('a');
                                        link.href = dataUrl;
                                        link.download = 'protein_interface_3d_view.png';
                                        document.body.appendChild(link);
                                        link.click();
                                        document.body.removeChild(link);
                                    }} catch(err) {{
                                        alert('Failed to generate PNG capture: ' + err.message);
                                    }}
                                }});
                            </script>
                            """
                            components.html(py3dmol_html, height=570)
                        except Exception as e:
                            st.error(f"Could not render 3D structural model: {e}")

                # --- Inter-chain Contacts Results Table Header ---
                st.subheader(f"Inter-chain Contacts: Chain {chain_a} and Chain {chain_b} at {selected_cutoff} Å")

                # Display Metrics Dashboard
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_unique_res = len(interface_residues_a) + len(interface_residues_b)
                    st.metric("Unique Interface Residues", f"{total_unique_res} (C1: {len(interface_residues_a)}, C2: {len(interface_residues_b)})")
                with col2:
                    st.metric("Residue–Residue Pairs", len(df_pairs))
                with col3:
                    st.metric("Atom–Atom Contacts", len(df_atoms))

                if not df_pairs.empty:
                    # Table data tabs (excluding Help Guide as it has been moved to top navigation)
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

                    # PDF Generation Export Section
                    st.markdown("---")
                    st.subheader("📥 Export Report as PDF")
                    st.write("Generate a detailed, formatted PDF summary containing structural metrics, amino acid category breakdowns, the contact distance plot, and the contact residue table.")

                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
                    styles = getSampleStyleSheet()

                    # Define Custom Styles safely
                    title_style = ParagraphStyle(
                        'ReportTitleStyle',
                        parent=styles['Heading1'],
                        fontSize=22,
                        textColor=colors.HexColor('#1E3A8A'),
                        spaceAfter=15
                    )
                    section_style = ParagraphStyle(
                        'ReportSectionStyle',
                        parent=styles['Heading2'],
                        fontSize=14,
                        textColor=colors.HexColor('#2563EB'),
                        spaceBefore=12,
                        spaceAfter=8
                    )
                    normal_style = styles['Normal']

                    story = []
                    # Header
                    story.append(Paragraph("Protein Contact Residue Server Analysis Report", title_style))
                    story.append(Paragraph(f"<b>Structure Target:</b> {pdb_id.upper()}", normal_style))
                    story.append(Paragraph(f"<b>Interface Chains:</b> Chain {chain_a} & Chain {chain_b}", normal_style))
                    story.append(Paragraph(f"<b>Distance Cut-off Threshold:</b> {selected_cutoff} Å", normal_style))
                    story.append(Spacer(1, 15))

                    # Metrics Section
                    story.append(Paragraph("1. Key Interface Metrics", section_style))
                    metrics_table_data = [
                        ["Metric Name", "Value"],
                        ["Total Unique Interface Residues", str(total_unique_res)],
                        [f"- Chain {chain_a} Residues", str(len(interface_residues_a))],
                        [f"- Chain {chain_b} Residues", str(len(interface_residues_b))],
                        ["Residue-Residue Pairs", str(len(df_pairs))],
                        ["Atom-Atom Contacts", str(len(df_atoms))]
                    ]
                    t_metrics = Table(metrics_table_data, colWidths=[250, 100])
                    t_metrics.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#3B82F6')),
                        ('TEXTCOLOR', (0,0), (1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#E5E7EB'))
                    ]))
                    story.append(t_metrics)
                    story.append(Spacer(1, 15))

                    # Property Analysis Section
                    if df_groups is not None:
                        story.append(Paragraph("2. Chemical Group Distributions", section_style))
                        groups_table_data = [["Category", "Count", "Percentage (%)"]]
                        for idx, row in df_groups.iterrows():
                            groups_table_data.append([str(row["Category"]), str(row["Count"]), f"{row['Percentage (%)']:.2f}%"])
                        t_groups = Table(groups_table_data, colWidths=[200, 70, 100])
                        t_groups.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#10B981')),
                            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                            ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#E5E7EB'))
                        ]))
                        story.append(t_groups)
                        story.append(Spacer(1, 15))

                    # Distance Distribution Plot Section
                    if not df_pairs.empty and "Minimum Distance (Å)" in df_pairs.columns:
                        story.append(Paragraph("3. Interface Contact Distance Summary Plot", section_style))
                        try:
                            fig, ax = plt.subplots(figsize=(6, 3))
                            distances = df_pairs["Minimum Distance (Å)"].dropna().astype(float)
                            ax.hist(distances, bins=10, color="#3B82F6", edgecolor="black", alpha=0.8)
                            ax.set_title("Distribution of Minimum Residue-Residue Distances", fontsize=10, fontweight="bold", color="#1E3A8A")
                            ax.set_xlabel("Distance (Å)", fontsize=8)
                            ax.set_ylabel("Frequency", fontsize=8)
                            ax.grid(True, linestyle="--", alpha=0.5)
                            plt.tight_layout()

                            img_buf = io.BytesIO()
                            plt.savefig(img_buf, format="png", dpi=150)
                            img_buf.seek(0)
                            plt.close(fig)

                            story.append(Image(img_buf, width=360, height=180))
                            story.append(Spacer(1, 15))
                        except Exception as ex_plot:
                            story.append(Paragraph(f"<i>* Note: Could not render interface contact plot: {ex_plot}</i>", normal_style))
                            story.append(Spacer(1, 15))

                    # 4. Residue-Residue Contacts Table Section inside PDF
                    story.append(Paragraph("4. Residue-Residue Contact Pairs Table", section_style))
                    pairs_table_data = [["S.No", "ResA", "ResB", "Distance (Å)"]]

                    # Include rows up to 100 for safety, or all if smaller
                    max_rows_to_render = min(100, len(df_pairs))
                    for idx, row in df_pairs.head(max_rows_to_render).iterrows():
                        res_a_str = f"{row['Residue A Name']}{row['Residue A ID']}"
                        res_b_str = f"{row['Residue B Name']}{row['Residue B ID']}"
                        try:
                            dist_val = float(row["Minimum Distance (Å)"])
                            dist_formatted = f"{dist_val:.3f}"
                        except:
                            dist_formatted = str(row["Minimum Distance (Å)"])

                        pairs_table_data.append([
                            str(idx),
                            res_a_str,
                            res_b_str,
                            dist_formatted
                        ])

                    # Dynamic Column Width adjustment for report flow
                    t_pairs = Table(pairs_table_data, colWidths=[60, 110, 110, 120])
                    t_pairs.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F3F4F6')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,-1), 9),
                        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#E5E7EB')),
                        ('TOPPADDING', (0,0), (-1,-1), 5),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ]))
                    story.append(t_pairs)

                    if len(df_pairs) > max_rows_to_render:
                        story.append(Spacer(1, 5))
                        story.append(Paragraph(f"<i>* Note: Only the first {max_rows_to_render} contacts are displayed here. Download the complete CSV for full contact sets.</i>", normal_style))

                    # Build document
                    doc.build(story)
                    pdf_data = pdf_buffer.getvalue()
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name=f"protein_contact_report_{pdb_id}_{chain_a}_{chain_b}.pdf",
                        mime="application/pdf"
                    )
    else:
        st.info("👈 Please load a structure from the sidebar to begin.")
