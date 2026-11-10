

import streamlit as st
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, PDBList
import io
import os
import streamlit.components.v1 as components

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

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
st.markdown('<div class="main-title">Protein Contact Residue Server</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4B5563; font-size: 1.15rem; margin-bottom: 25px;'>Analyze, visualize, and export residue contacts at protein-protein interfaces in 3D.</p>", unsafe_allow_html=True)

# Helper to download PDB structures
def get_pdb_file(pdb_id):
    pdbl = PDBList()
    # Fetch to current directory
    file_path = pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="pdb")
    if os.path.exists(file_path):
        return file_path
    else:
        # Try alternative naming standard
        alt_path = f"pdb{pdb_id.lower()}.ent"
        if os.path.exists(alt_path):
            return alt_path
    return None

# Side bar inputs
st.sidebar.markdown('<div class="sidebar-header-highlight">Configure Parameters</div>', unsafe_allow_html=True)

input_source = st.sidebar.radio("PDB Input Source:", ["Download PDB by ID", "Upload PDB File"])
pdb_code = ""
uploaded_file = None

if input_source == "Download PDB by ID":
    pdb_code = st.sidebar.text_input("Enter 4-Character PDB Code:", value="1Y26").strip().upper()
else:
    uploaded_file = st.sidebar.file_uploader("Upload PDB File:", type=["pdb", "ent"])

chain_a_input = st.sidebar.text_input("Chain A Name:", value="A").strip()
chain_b_input = st.sidebar.text_input("Chain B Name:", value="B").strip()
cutoff = st.sidebar.slider("Heavy Atom Distance Threshold (Å):", min_value=2.0, max_value=8.0, value=5.0, step=0.5)

st.sidebar.markdown('<div class="sidebar-header-highlight">Visual Preferences</div>', unsafe_allow_html=True)
bg_color_option = st.sidebar.selectbox("3D Canvas Background Color:", ["black", "white", "#1e1e1e", "lightgray"])
chain_a_color = st.sidebar.color_picker("Chain A Cartoon Color:", "#3B82F6")
chain_b_color = st.sidebar.color_picker("Chain B Cartoon Color:", "#F59E0B")

# Setup parsing and analysis
pdb_filepath = None
if input_source == "Download PDB by ID" and pdb_code:
    if len(pdb_code) == 4:
        with st.spinner(f"Downloading PDB: {pdb_code}..."):
            pdb_filepath = get_pdb_file(pdb_code)
            if not pdb_filepath:
                st.error("PDB file could not be downloaded. Verify the 4-digit code.")
    else:
        st.warning("Please enter a valid 4-character PDB ID.")
elif input_source == "Upload PDB File" and uploaded_file is not None:
    # Save uploaded file temporarily
    pdb_filepath = "uploaded_structure.pdb"
    with open(pdb_filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

if pdb_filepath:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_filepath)
    except Exception as e:
        st.error(f"Failed to parse PDB structure: {e}")
        st.stop()

    model = structure[0]

    # Validate user chain inputs
    available_chains = [c.id for c in model.get_chains()]
    if chain_a_input not in available_chains or chain_b_input not in available_chains:
        st.error(f"Invalid Chains specified. Available chains in this PDB structure are: {', '.join(available_chains)}")
        st.stop()

    chain_a = model[chain_a_input]
    chain_b = model[chain_b_input]

    # Computation of Contact residues
    contact_pairs = []
    unique_res_a = set()
    unique_res_b = set()

    for residue_a in chain_a:
        # Ensure it's standard residue, skip heteroatoms/water if preferred
        if residue_a.id[0] != " ":
            continue
        for residue_b in chain_b:
            if residue_b.id[0] != " ":
                continue
            
            # Compare heavy atoms
            found_contact = False
            min_dist = 999.0
            best_atom_a = ""
            best_atom_b = ""
            for atom_a in residue_a:
                if atom_a.element == "H":
                    continue
                for atom_b in residue_b:
                    if atom_b.element == "H":
                        continue
                    dist = atom_a - atom_b
                    if dist < min_dist:
                        min_dist = dist
                        best_atom_a = atom_a.get_name()
                        best_atom_b = atom_b.get_name()
                    if dist <= cutoff:
                        found_contact = True

            if found_contact:
                contact_pairs.append({
                    "Residue A": f"{residue_a.get_resname()}{residue_a.id[1]}",
                    "Residue B": f"{residue_b.get_resname()}{residue_b.id[1]}",
                    "Min Distance (Å)": round(min_dist, 3),
                    "Atom A": best_atom_a,
                    "Atom B": best_atom_b
                })
                unique_res_a.add(f"{residue_a.get_resname()}{residue_a.id[1]}")
                unique_res_b.add(f"{residue_b.get_resname()}{residue_b.id[1]}")

    df_contacts = pd.DataFrame(contact_pairs)

    # Create Interface tab panels
    tab_results, tab_metrics, tab_viz = st.tabs([
        "📊 Contact Results Table", 
        "📝 Interface Metrics Reference", 
        "🧬 3D Interactive Canvas"
    ])

    with tab_results:
        st.subheader("Discovered Interface Contact Pairs")
        if not df_contacts.empty:
            st.dataframe(df_contacts, use_container_width=True)
            
            # Create PDF generator
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Heading1'],
                fontSize=22,
                textColor=colors.HexColor('#1E3A8A'),
                spaceAfter=15
            )
            meta_style = ParagraphStyle(
                'MetaStyle',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#4B5563'),
                spaceAfter=10
            )
            section_style = ParagraphStyle(
                'SectionStyle',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2563EB'),
                spaceAfter=8
            )
            
            story = []
            story.append(Paragraph("Protein Contact Residue Analysis Report", title_style))
            story.append(Paragraph(f"PDB Target: {pdb_code if input_source == 'Download PDB by ID' else 'Uploaded Structure'}", meta_style))
            story.append(Paragraph(f"Parameters: Chain A: {chain_a_input} | Chain B: {chain_b_input} | Cutoff: {cutoff}Å", meta_style))
            story.append(Spacer(1, 10))
            
            story.append(Paragraph("Summary Statistics", section_style))
            summary_data = [
                ["Metric Name", "Value"],
                ["Atom-Atom Contacts Found", str(len(df_contacts))],
                [f"Unique Interface Residues (Chain {chain_a_input})", str(len(unique_res_a))],
                [f"Unique Interface Residues (Chain {chain_b_input})", str(len(unique_res_b))]
            ]
            summary_table = Table(summary_data, colWidths=[250, 150])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3B82F6')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F9FAFB'), colors.white]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB'))
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 15))
            
            story.append(Paragraph("Residue Contact Pair Matrix Table", section_style))
            # Cap at first 50 entries to keep layout clean, let users check streamlit UI for full lists
            table_df = df_contacts.head(60)
            table_headers = ["Residue A", "Residue B", "Atom A", "Atom B", "Min Dist (Å)"]
            table_matrix = [table_headers] + table_df.values.tolist()
            
            contacts_table = Table(table_matrix, colWidths=[100, 100, 100, 100, 100])
            contacts_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('BOTTOMPADDING', (0,0), (-1,0), 5),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F3F4F6'), colors.white]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB'))
            ]))
            story.append(contacts_table)
            
            doc.build(story)
            pdf_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Complete PDF Analysis Report",
                data=pdf_buffer,
                file_name="protein_contact_residue_report.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No interface residues found under current spatial distance parameters.")

    with tab_metrics:
        st.subheader("Interface Metrics Structural Guideline")
        st.markdown("""
        ### Core Definitions
        * **Atom–Atom Contacts**: The total count of all individual heavy atom pairs within the cut-off distance.
        * **Residue–Residue Contact Pairs**: Count of distinct pair mappings between Chain A and Chain B. (Shown in table tab)
        * **Unique Interface Residues**: Count of singular unique protein residues active on either interface chain.
        """)

    with tab_viz:
        st.subheader("Interactive 3D Structure Viewer")
        if not df_contacts.empty:
            # Construct specific styling scripts
            with open(pdb_filepath, 'r') as f:
                escaped_pdb = f.read().replace('\n', '\\n').replace('\'', '\\\'').replace('"', '\\"')
            
            # Create styling blocks for py3dmol
            sidechains_js = ""
            labels_js = ""
            
            # Render Contact Sidechains in distinctive styles
            for res in unique_res_a:
                # Parse Residue number
                res_num = ''.join(filter(str.isdigit, res))
                sidechains_js += f"viewer.addStyle({{chain: '{chain_a_input}', resi: {res_num}}}, {{stick: {{colorscheme: 'cyanCarbon'}}}});\n"
                labels_js += f"viewer.addLabel('{res}', {{font: 'Arial', fontSize: 10, fontColor: 'white', backgroundOpacity: 0.5, backgroundColor: 'black'}}, {{chain: '{chain_a_input}', resi: {res_num}}});\n"
                
            for res in unique_res_b:
                res_num = ''.join(filter(str.isdigit, res))
                sidechains_js += f"viewer.addStyle({{chain: '{chain_b_input}', resi: {res_num}}}, {{stick: {{colorscheme: 'orangeCarbon'}}}});\n"
                labels_js += f"viewer.addLabel('{res}', {{font: 'Arial', fontSize: 10, fontColor: 'white', backgroundOpacity: 0.5, backgroundColor: 'black'}}, {{chain: '{chain_b_input}', resi: {res_num}}});\n"

            py3dmol_html = f"""
            <div id="3dmol_canvas" style="width: 100%; height: 500px; position: relative;"></div>
            <div style="margin-top: 10px; margin-bottom: 20px; display: flex; gap: 10px;">
                <button onclick="window.reset3DView()" style="
                    background-color: #6B7280;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-weight: 600;
                    cursor: pointer;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">𖤐 Reset 3D View</button>
                <button onclick="window.downloadPNG()" style="
                    background-color: #2563EB;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-weight: 600;
                    cursor: pointer;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">📸 Save 3D View as PNG Image</button>
                <button onclick="window.downloadJPEG()" style="
                    background-color: #10B981;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-weight: 600;
                    cursor: pointer;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">📸 Save 3D View as JPEG Image</button>
            </div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/1.7.0/3Dmol-min.js"></script>
            <script>
                var viewer;
                function initializeViewer() {{
                    var element = document.getElementById('3dmol_canvas');
                    if (!element) {{
                        setTimeout(initializeViewer, 100);
                        return;
                    }}
                    var config = {{ backgroundColor: '{bg_color_option}' }};
                    viewer = $3Dmol.createViewer(element, config);
                    viewer.addModel('{escaped_pdb}', 'pdb');

                    // Apply base styles
                    applyBaseStyles();
                    viewer.zoomTo();
                    viewer.render();
                }}

                function applyBaseStyles() {{
                    if (!viewer) return;
                    // Clear old styling and labels completely
                    viewer.setStyle({{}}, {{}});
                    viewer.clearLabels();

                    // Set Ribbon styles for Chains
                    viewer.setStyle({{chain: '{chain_a_input}'}}, {{cartoon: {{color: '{chain_a_color}'}}}});
                    viewer.setStyle({{chain: '{chain_b_input}'}}, {{cartoon: {{color: '{chain_b_color}'}}}});

                    // Draw sidechain sticks & contact labels
                    {sidechains_js}
                    {labels_js}
                }}

                // Bind helper functions globally to window context to guarantee button triggers work
                window.reset3DView = function() {{
                    if (viewer) {{
                        applyBaseStyles();
                        viewer.zoomTo();
                        viewer.render();
                    }}
                }};

                window.downloadPNG = function() {{
                    try {{
                        var canvas = document.getElementsByTagName('canvas')[0];
                        if (!canvas) throw new Error("Canvas element not found.");
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
                }};

                window.downloadJPEG = function() {{
                    try {{
                        var canvas = document.getElementsByTagName('canvas')[0];
                        if (!canvas) throw new Error("Canvas element not found.");
                        var dataUrl = canvas.toDataURL('image/jpeg', 0.9);
                        var link = document.createElement('a');
                        link.href = dataUrl;
                        link.download = 'protein_interface_3d_view.jpg';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    }} catch(err) {{
                        alert('Failed to generate JPEG capture: ' + err.message);
                    }}
                }};
                
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', initializeViewer);
                }} else {{
                    initializeViewer();
                }}
            </script>
            """
            components.html(py3dmol_html, height=600)
        else:
            st.info("No structures loaded under current filter criteria.")
