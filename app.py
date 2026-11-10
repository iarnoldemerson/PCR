
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

# Sidebar options
st.sidebar.markdown('<div class="sidebar-header-highlight">🧬 Input Options</div>', unsafe_allow_html=True)
pdb_source = st.sidebar.radio("Select PDB Source", ["Fetch from PDB", "Upload PDB File"])

pdb_code = ""
uploaded_file = None
if pdb_source == "Fetch from PDB":
    pdb_code = st.sidebar.text_input("Enter 4-letter PDB Code", value="1A8G").strip().upper()
else:
    uploaded_file = st.sidebar.file_uploader("Upload PDB File", type=["pdb", "ent"])

distance_threshold = st.sidebar.slider("Distance Threshold (Å)", min_value=2.0, max_value=10.0, value=5.0, step=0.5)

def load_pdb_structure(source, code, file_obj):
    parser = PDBParser(QUIET=True)
    if source == "Fetch from PDB" and code:
        pdbl = PDBList()
        filename = pdbl.retrieve_pdb_file(code, pdir=".", file_format="pdb")
        if os.path.exists(filename):
            return parser.get_structure(code, filename), filename
    elif source == "Upload PDB File" and file_obj is not None:
        stringio = io.StringIO(file_obj.getvalue().decode("utf-8"))
        temp_path = "uploaded_temp.pdb"
        with open(temp_path, "w") as f:
            f.write(stringio.read())
        return parser.get_structure("uploaded", temp_path), temp_path
    return None, None

struct, struct_path = load_pdb_structure(pdb_source, pdb_code, uploaded_file)

if struct:
    # Extract chains dynamically
    model = struct[0]
    chains = [chain.id for chain in model]
    
    st.sidebar.markdown('<div class="sidebar-header-highlight">🔗 Select Chains</div>', unsafe_allow_html=True)
    chain_a_name = st.sidebar.selectbox("Chain A", chains, index=0)
    chain_b_name = st.sidebar.selectbox("Chain B", chains, index=min(1, len(chains)-1))
    
    if chain_a_name == chain_b_name:
        st.sidebar.warning("Warning: Selecting the same chain for both A and B.")
        
    chain_a = model[chain_a_name]
    chain_b = model[chain_b_name]
    
    # Compute contacts
    contacts = []
    for res_a in chain_a:
        if res_a.id[0] != " ": continue  # skip heteroatoms
        for res_b in chain_b:
            if res_b.id[0] != " ": continue
            
            min_dist = float('inf')
            for atom_a in res_a:
                for atom_b in res_b:
                    dist = np.linalg.norm(atom_a.coord - atom_b.coord)
                    if dist < min_dist:
                        min_dist = dist
            
            if min_dist <= distance_threshold:
                label_a = f"{res_a.get_resname()}{res_a.id[1]}"
                label_b = f"{res_b.get_resname()}{res_b.id[1]}"
                contacts.append({
                    "Chain A Residue": label_a,
                    "Chain B Residue": label_b,
                    "Distance (Å)": round(min_dist, 3),
                    "spec_a": {"chain": chain_a_name, "resi": res_a.id[1]},
                    "spec_b": {"chain": chain_b_name, "resi": res_b.id[1]},
                    "label_a": label_a,
                    "label_b": label_b
                })
                
    df_contacts = pd.DataFrame(contacts)
    
    tab1, tab2, tab3 = st.tabs(["📊 Contact Details & Stats", "🌐 3D Structural View", "📝 Interface Reference Guide"])
    
    with tab1:
        st.subheader("Contact Summary Metrics")
        col1, col2, col3 = st.columns(3)
        
        if not df_contacts.empty:
            col1.metric("Contact Pairs Found", len(df_contacts))
            
            all_unique_res = list(set(df_contacts["Chain A Residue"].tolist() + df_contacts["Chain B Residue"].tolist()))
            col2.metric("Unique Interface Residues", len(all_unique_res))
            col3.metric("Avg Distance", f"{df_contacts['Distance (Å)'].mean():.2f} Å")
            
            st.markdown("### Contact Residues DataFrame")
            st.dataframe(df_contacts[["Chain A Residue", "Chain B Residue", "Distance (Å)"]], use_container_width=True)
            
            # Calculate statistics
            df_aa, df_group, aa_counts = get_amino_acid_statistics(all_unique_res)
            
            st.markdown("### Amino Acid Profile at Interface")
            c_left, c_right = st.columns(2)
            with c_left:
                st.write("**Amino Acid Distributions**")
                st.dataframe(df_aa, use_container_width=True)
            with c_right:
                st.write("**Operational Property Classification**")
                st.dataframe(df_group, use_container_width=True)
        else:
            st.info("No contacts found at current distance threshold.")
            
    with tab2:
        st.subheader("3D Viewer (py3Dmol with Selected Interactions Labels)")
        if not df_contacts.empty:
            # Read structure text for JS injector
            with open(struct_path, "r") as f:
                pdb_data_str = f.read().replace("\n", "\\n").replace("'", "\\'").replace('"', '\\"')
            
            # Format JavaScript calls for selected top contacts
            labels_js = ""
            for idx, link in df_contacts.head(10).iterrows():
                labels_js += f"""Viewer.addLabel('{link["label_a"]}', {{{{fontColor:'black', backgroundColor:'#DCE7F3', fontSize:10, backgroundOpacity:0.8, borderThickness:1.0, borderColor:'#3B82F6'}}}}, {{{{chain: '{link["spec_a"]["chain"]}', resi: {link["spec_a"]["resi"]}}}}});\n"""
                labels_js += f"""Viewer.addLabel('{link["label_b"]}', {{{{fontColor:'black', backgroundColor:'#FEE2E2', fontSize:10, backgroundOpacity:0.8, borderThickness:1.0, borderColor:'#EF4444'}}}}, {{{{chain: '{link["spec_b"]["chain"]}', resi: {link["spec_b"]["resi"]}}}}});\n"""
            
            html_code = f"""
            <div id="container" style="height: 500px; width: 100%; position: relative;"></div>
            <script src="https://3dmol.org/build/3Dmol-min.js"></script>
            <script>
                let element = document.getElementById('container');
                let config = {{defaultcolors: $3Dmol.rasmolElementColors}};
                let Viewer = $3Dmol.createViewer(element, config);
                Viewer.addModel(`{pdb_data_str}`, "pdb");
                Viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                Viewer.zoomTo();
                
                // Add custom interface contact visualizers
                {labels_js}
                
                Viewer.render();
            </script>
            """
            components.html(html_code, height=520)
        else:
            st.info("Visualize 3D structure by selecting chains with active contact points.")
            
    with tab3:
        st.markdown("""
        ### Interface Metrics Reference Guidelines
        Ensure you are accurately interpreting the difference between the following parameters:
        
        1. **Number of Atom–Atom Contacts**:
           * The total frequency count of individual atom pairs below the threshold distance.
           
        2. **Number of Unique Residue–Residue Contact Pairs**:
           * Number of unique pairs of interacting residues (e.g. Chain A Residue i - Chain B Residue j).
           
        3. **Number of Unique Interface Residues**:
           * Total unique set of residues that participate on either side of the interface.
        """)
else:
    st.info("Please choose a valid PDB source on the sidebar to trigger analysis.")
