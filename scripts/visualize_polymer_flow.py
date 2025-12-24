#!/usr/bin/env python
"""
Interactive visualization of PolymerFlowModel latent space.

Visualize multiple chained residues with sliders controlling each residue's
latent vector. The 3D structure updates in real-time.

Usage:
    python scripts/visualize_polymer_flow.py
    python scripts/visualize_polymer_flow.py --n-residues 5 --n-structures 200
"""

import numpy as np
import torch
from pathlib import Path

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

from ciffy.biochemistry import Residue
from ciffy.nn.flow import (
    ResidueFlowModel,
    ResidueFlowConfig,
    PolymerFlowModel,
)
from ciffy.nn.flow.residue.data import extract_residues_with_links, compute_pca


# Atom colors by element
ATOM_COLORS = {
    'C': '#909090',
    'N': '#3050F8',
    'O': '#FF0D0D',
    'P': '#FF8000',
    'H': '#FFFFFF',
}


def create_newton_projector(atoms: list, residue: Residue, n_steps: int = 2):
    """
    Create a Newton projection function for bond length and angle constraints.

    Returns a function that projects coordinates onto geometric constraints.
    """
    # Reference bond lengths
    ref_bonds = [
        (residue.P, residue.OP1, 1.484),
        (residue.P, residue.OP2, 1.484),
        (residue.P, residue.O5p, 1.594),
        (residue.O5p, residue.C5p, 1.430),
        (residue.C5p, residue.C4p, 1.512),
        (residue.C4p, residue.C3p, 1.520),
        (residue.C3p, residue.O3p, 1.423),
        (residue.C4p, residue.O4p, 1.448),
        (residue.O4p, residue.C1p, 1.418),
        (residue.C1p, residue.C2p, 1.529),
        (residue.C2p, residue.C3p, 1.523),
        (residue.C1p, residue.N9, 1.465),
    ]

    # Reference angles (atom_i, atom_j, atom_k, angle_in_degrees)
    # Angle is at atom_j between i-j-k
    ref_angles = [
        (residue.OP1, residue.P, residue.OP2, 119.494),
        (residue.O5p, residue.P, residue.OP1, 108.365),
        (residue.O5p, residue.P, residue.OP2, 108.246),
    ]

    atom_to_idx = {a: i for i, a in enumerate(atoms)}
    n_atoms = len(atoms)

    # Build bond constraint data
    bond_pairs = []
    bond_targets = []
    for a1, a2, target in ref_bonds:
        if a1.value in atom_to_idx and a2.value in atom_to_idx:
            bond_pairs.append((atom_to_idx[a1.value], atom_to_idx[a2.value]))
            bond_targets.append(target)
    bond_pairs = torch.tensor(bond_pairs)
    bond_targets = torch.tensor(bond_targets, dtype=torch.float32)
    n_bonds = len(bond_targets)

    # Build angle constraint data
    # constraint: (x_i - x_j) · (x_k - x_j) = cos(θ) * d_ij * d_jk
    angle_triples = []
    angle_targets = []  # cos(θ) * d_ij * d_jk
    bond_length_map = {(a1.value, a2.value): d for a1, a2, d in ref_bonds}
    bond_length_map.update({(a2.value, a1.value): d for a1, a2, d in ref_bonds})

    for a_i, a_j, a_k, angle_deg in ref_angles:
        if all(a.value in atom_to_idx for a in [a_i, a_j, a_k]):
            i, j, k = atom_to_idx[a_i.value], atom_to_idx[a_j.value], atom_to_idx[a_k.value]
            angle_triples.append((i, j, k))
            # Get bond lengths
            d_ij = bond_length_map.get((a_i.value, a_j.value), 1.5)
            d_jk = bond_length_map.get((a_j.value, a_k.value), 1.5)
            cos_theta = np.cos(np.radians(angle_deg))
            angle_targets.append(cos_theta * d_ij * d_jk)

    angle_triples = torch.tensor(angle_triples)
    angle_targets = torch.tensor(angle_targets, dtype=torch.float32)
    n_angles = len(angle_targets)

    def newton_step(coords):
        """Single Newton step for bonds + angles. coords: (n_atoms, 3)"""
        n_constraints = n_bonds + n_angles
        residuals = torch.zeros(n_constraints)
        J = torch.zeros(n_constraints, n_atoms * 3)

        # Bond length constraints
        for idx, (a1, a2) in enumerate(bond_pairs):
            diff = coords[a2] - coords[a1]
            length = torch.norm(diff)
            residuals[idx] = length - bond_targets[idx]
            unit = diff / (length + 1e-8)
            J[idx, a1*3:a1*3+3] = -unit
            J[idx, a2*3:a2*3+3] = unit

        # Angle constraints: (x_i - x_j) · (x_k - x_j) = target
        for idx, (i, j, k) in enumerate(angle_triples):
            v1 = coords[i] - coords[j]  # x_i - x_j
            v2 = coords[k] - coords[j]  # x_k - x_j
            dot_product = torch.dot(v1, v2)
            residuals[n_bonds + idx] = dot_product - angle_targets[idx]
            # Jacobian: d(v1·v2)/d(x_i) = v2, d(v1·v2)/d(x_k) = v1, d(v1·v2)/d(x_j) = -v1 - v2
            J[n_bonds + idx, i*3:i*3+3] = v2
            J[n_bonds + idx, k*3:k*3+3] = v1
            J[n_bonds + idx, j*3:j*3+3] = -v1 - v2

        # Gauss-Newton: dx = -J^T @ (J @ J^T)^{-1} @ residuals
        JJT = J @ J.T
        y = torch.linalg.solve(JJT, residuals)
        dx = -J.T @ y

        return coords + dx.reshape(n_atoms, 3)

    def project(coords):
        """Apply n_steps Newton steps. coords: (n_atoms, 3) numpy array"""
        c = torch.from_numpy(coords.astype(np.float32))
        for _ in range(n_steps):
            c = newton_step(c)
        return c.numpy()

    return project

# Residue colors for distinguishing residues
RESIDUE_COLORS = [
    '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00',
    '#FFFF33', '#A65628', '#F781BF', '#999999', '#66C2A5',
]


def get_residue_bonds(atoms: list, residue: Residue) -> list:
    """Get bond connectivity from residue topology."""
    atom_to_col = {a: i for i, a in enumerate(atoms)}
    bonds = []
    for i, j in residue.bond_indices:
        if i in atom_to_col and j in atom_to_col:
            bonds.append((atom_to_col[i], atom_to_col[j]))
    return bonds


def get_atom_element(atom_idx: int, residue: Residue) -> str:
    """Get element symbol for an atom index."""
    for attr_name in dir(residue):
        attr = getattr(residue, attr_name)
        if hasattr(attr, 'value') and attr.value == atom_idx:
            if attr_name.startswith('O'):
                return 'O'
            elif attr_name.startswith('N'):
                return 'N'
            elif attr_name.startswith('C'):
                return 'C'
            elif attr_name.startswith('P'):
                return 'P'
            elif attr_name.startswith('H'):
                return 'H'
    return 'C'


def compute_geometry_stats(
    coords: np.ndarray,
    atoms: list,
    residue: Residue,
) -> dict:
    """Compute key bond lengths and angles for a residue."""
    atom_to_col = {a: i for i, a in enumerate(atoms)}

    def get_col(attr_name):
        attr = getattr(residue, attr_name)
        return atom_to_col.get(attr.value)

    def bond_length(a1, a2):
        c1, c2 = get_col(a1), get_col(a2)
        if c1 is None or c2 is None:
            return None
        return np.linalg.norm(coords[c1] - coords[c2])

    def angle(a1, a2, a3):
        """Angle at a2 between a1-a2-a3."""
        c1, c2, c3 = get_col(a1), get_col(a2), get_col(a3)
        if c1 is None or c2 is None or c3 is None:
            return None
        v1 = coords[c1] - coords[c2]
        v2 = coords[c3] - coords[c2]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    return {
        # Phosphate bonds
        'P-OP1': bond_length('P', 'OP1'),
        'P-OP2': bond_length('P', 'OP2'),
        'P-O5\'': bond_length('P', 'O5p'),
        # Sugar bonds
        'C1\'-N9': bond_length('C1p', 'N9'),
        'C1\'-O4\'': bond_length('C1p', 'O4p'),
        'C1\'-C2\'': bond_length('C1p', 'C2p'),
        # Phosphate angles
        'OP1-P-OP2': angle('OP1', 'P', 'OP2'),
        'O5\'-P-OP1': angle('O5p', 'P', 'OP1'),
        'O5\'-P-OP2': angle('O5p', 'P', 'OP2'),
    }


def compute_reference_geometry(coords_all: np.ndarray, atoms: list, residue: Residue) -> dict:
    """Compute mean and std of geometry from training data."""
    all_stats = []
    for i in range(len(coords_all)):
        stats = compute_geometry_stats(coords_all[i], atoms, residue)
        all_stats.append(stats)

    ref = {}
    for key in all_stats[0].keys():
        values = [s[key] for s in all_stats if s[key] is not None]
        if values:
            ref[key] = {'mean': np.mean(values), 'std': np.std(values)}
    return ref


def create_polymer_figure(
    all_coords: list[np.ndarray],
    bonds: list,
    atom_colors: list,
    atoms: list,
    residue: Residue,
    selected_idx: int = 0,
):
    """Create 3D molecular figure with multiple residues."""
    fig = go.Figure()

    n_residues = len(all_coords)
    o3p_col = atoms.index(residue.O3p.value)
    p_col = atoms.index(residue.P.value)

    for res_idx, coords in enumerate(all_coords):
        is_selected = (res_idx == selected_idx)
        base_color = RESIDUE_COLORS[res_idx % len(RESIDUE_COLORS)]

        # Bonds
        x_bonds, y_bonds, z_bonds = [], [], []
        for i, j in bonds:
            x_bonds.extend([coords[i, 0], coords[j, 0], None])
            y_bonds.extend([coords[i, 1], coords[j, 1], None])
            z_bonds.extend([coords[i, 2], coords[j, 2], None])

        fig.add_trace(go.Scatter3d(
            x=x_bonds, y=y_bonds, z=z_bonds,
            mode='lines',
            line=dict(color=base_color, width=6 if is_selected else 4),
            name=f'Bonds {res_idx+1}',
            hoverinfo='skip',
            showlegend=False,
        ))

        # Atoms
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=10 if is_selected else 7,
                color=atom_colors,
                opacity=1.0 if is_selected else 0.7,
                line=dict(width=2 if is_selected else 1, color=base_color),
            ),
            name=f'Residue {res_idx+1}',
            hoverinfo='text',
            text=[f'Res {res_idx+1}' for _ in range(len(coords))],
            showlegend=True,
        ))

        # Backbone link to next residue
        if res_idx < n_residues - 1:
            next_coords = all_coords[res_idx + 1]
            o3p = coords[o3p_col]
            p_next = next_coords[p_col]

            fig.add_trace(go.Scatter3d(
                x=[o3p[0], p_next[0]],
                y=[o3p[1], p_next[1]],
                z=[o3p[2], p_next[2]],
                mode='lines',
                line=dict(color='orange', width=8),
                name=f'Link {res_idx+1}-{res_idx+2}',
                hoverinfo='skip',
                showlegend=False,
            ))

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            bgcolor='white',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
        title=dict(text=f'PolymerFlow ({n_residues} Residues)', x=0.5),
    )

    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8052)
    parser.add_argument('--n-structures', type=int, default=100)
    parser.add_argument('--latent-dim', type=int, default=12)
    parser.add_argument('--n-residues', type=int, default=3)
    parser.add_argument('--n-epochs', type=int, default=100)
    args = parser.parse_args()

    n_residues = args.n_residues
    latent_dim = args.latent_dim

    print(f"Training ResidueFlowModel for adenosine...")
    data_dir = Path("/Users/hmblair/academic/data/rna-libraries/pdb/structures/pdb130")
    paths = sorted(data_dir.glob("*.cif"))[:args.n_structures]

    # Train with standard MLE (geometry correction via Newton projection post-hoc)
    config = ResidueFlowConfig(
        latent_dim=latent_dim,
        n_layers=8,
        hidden_dim=64,
        min_coverage=0.9,
    )

    residue_model = ResidueFlowModel.from_structures(
        paths,
        Residue.A,
        config=config,
        n_epochs=args.n_epochs,
        verbose=True,
    )

    # Extract data for latent statistics
    coords_all, transforms_all, atoms = extract_residues_with_links(
        paths, Residue.A, min_coverage=0.9, verbose=False
    )

    print(f"\nResidueFlowModel: {residue_model}")

    # Create PolymerFlowModel
    polymer_model = PolymerFlowModel({Residue.A: residue_model})
    print(f"PolymerFlowModel: {polymer_model}")

    # Get atom info (already have atoms from extraction above)
    n_atoms = residue_model.n_atoms
    residue = residue_model.residue

    # Compute latent statistics from training data (already extracted above)
    coords_flat = coords_all.reshape(len(coords_all), -1)
    extended = np.concatenate([coords_flat, transforms_all], axis=1)
    X = torch.from_numpy(extended.astype(np.float32))

    with torch.no_grad():
        Z = residue_model.flow.encode(X).numpy()
    z_mean = Z.mean(axis=0)
    z_std = Z.std(axis=0)

    print(f"\nLatent stats: mean~{z_mean.round(2)}, std~{z_std.round(2)}")

    # Compute variance per component from PCA
    _, _, singular_values, var_explained = compute_pca(extended, n_components=latent_dim)
    total_var = (singular_values ** 2).sum()
    var_per_component = (singular_values ** 2) / total_var * 100

    # Compute reference geometry from training data
    ref_geometry = compute_reference_geometry(coords_all, atoms, residue)
    print("\nReference geometry (from training data):")
    for key, val in ref_geometry.items():
        unit = "Å" if "-" in key and "P-" not in key.split("-")[0] or key.count("-") == 1 else "°"
        if key.count("-") == 2:  # It's an angle
            unit = "°"
        else:
            unit = "Å"
        print(f"  {key}: {val['mean']:.3f} ± {val['std']:.3f} {unit}")

    # Bonds and colors
    bonds = get_residue_bonds(atoms, residue)
    atom_colors = [ATOM_COLORS.get(get_atom_element(a, residue), '#909090') for a in atoms]

    # Newton projector for geometry correction
    newton_project = create_newton_projector(atoms, residue, n_steps=2)

    # Create Dash app
    app = Dash(__name__)
    slider_range = 3.0

    # Sliders for latent dimensions
    sliders = [
        html.Div([
            html.Label(f"z{i+1}",
                      style={'fontWeight': 'bold', 'fontSize': '12px'}),
            dcc.Slider(
                id=f'slider-{i}',
                min=-slider_range, max=slider_range, step=0.1, value=0,
                marks={-3: '-3', 0: '0', 3: '+3'},
                tooltip={'placement': 'bottom'},
                updatemode='drag'
            ),
        ], style={'padding': '8px'}) for i in range(latent_dim)
    ]

    # Residue selector dropdown
    residue_options = [{'label': f'Residue {i+1}', 'value': i} for i in range(n_residues)]

    app.layout = html.Div([
        html.H2("PolymerFlow Visualization",
                style={'textAlign': 'center'}),
        html.P(f"Model: {latent_dim}D latent, {residue_model.var_explained*100:.1f}% var, "
               f"RMSD={residue_model.pca_rmsd:.3f}A | {n_residues} residues",
               style={'textAlign': 'center', 'color': 'gray'}),

        html.Div([
            html.Div([
                dcc.Graph(id='structure-figure', style={'height': '700px'}),
            ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.H4("Select Residue", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='residue-selector',
                    options=residue_options,
                    value=0,
                    clearable=False,
                    style={'marginBottom': '15px'}
                ),

                html.H4("Latent Space", style={'textAlign': 'center'}),
                *sliders,

                html.Div([
                    html.Button('Reset Selected', id='reset-button', n_clicks=0,
                               style={'width': '48%', 'padding': '10px', 'marginTop': '15px'}),
                    html.Button('Reset All', id='reset-all-button', n_clicks=0,
                               style={'width': '48%', 'padding': '10px', 'marginTop': '15px',
                                      'marginLeft': '4%'}),
                ]),

                html.Div([
                    dcc.Checklist(
                        id='newton-toggle',
                        options=[{'label': ' Newton projection (2 steps)', 'value': 'on'}],
                        value=['on'],  # On by default
                        style={'marginTop': '15px'}
                    ),
                ]),

                html.Div(id='info-text', style={'padding': '10px', 'fontSize': '11px',
                                                'marginTop': '15px', 'backgroundColor': '#fff',
                                                'borderRadius': '5px'}),
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'padding': '15px', 'backgroundColor': '#f5f5f5', 'borderRadius': '5px'}),
        ]),

        # Store for all residue latent vectors
        dcc.Store(id='latent-store', data={
            'latents': [[0.0] * latent_dim for _ in range(n_residues)]
        }),
    ], style={'fontFamily': 'Arial, sans-serif'})

    @app.callback(
        Output('latent-store', 'data'),
        Output('structure-figure', 'figure'),
        Output('info-text', 'children'),
        *[Output(f'slider-{i}', 'value') for i in range(latent_dim)],
        Input('residue-selector', 'value'),
        *[Input(f'slider-{i}', 'value') for i in range(latent_dim)],
        Input('reset-button', 'n_clicks'),
        Input('reset-all-button', 'n_clicks'),
        Input('newton-toggle', 'value'),
        State('latent-store', 'data'),
    )
    def update(selected_res, *args):
        from dash import ctx

        slider_vals = list(args[:latent_dim])
        reset_clicks = args[latent_dim]
        reset_all_clicks = args[latent_dim + 1]
        newton_enabled = 'on' in (args[latent_dim + 2] or [])
        store_data = args[latent_dim + 3]

        latents = [list(z) for z in store_data['latents']]

        # Handle reset buttons
        if ctx.triggered_id == 'reset-button':
            slider_vals = [0.0] * latent_dim
            latents[selected_res] = [0.0] * latent_dim
        elif ctx.triggered_id == 'reset-all-button':
            slider_vals = [0.0] * latent_dim
            latents = [[0.0] * latent_dim for _ in range(n_residues)]
        elif ctx.triggered_id == 'residue-selector':
            slider_vals = latents[selected_res]
        else:
            latents[selected_res] = slider_vals

        # Build latent tensor (sliders directly control z values)
        latent_array = np.array(latents)  # (n_residues, latent_dim)
        latent_tensor = torch.from_numpy(latent_array.astype(np.float32))

        # Decode using PolymerFlowModel
        sequence = [Residue.A] * n_residues
        with torch.no_grad():
            coords_flat = polymer_model.decode(latent_tensor, sequence)

        # Split back into per-residue coords for visualization
        all_coords = []
        offset = 0
        for _ in range(n_residues):
            coords_i = coords_flat[offset:offset + n_atoms].numpy()
            # Apply Newton projection if enabled
            if newton_enabled:
                coords_i = newton_project(coords_i)
            all_coords.append(coords_i)
            offset += n_atoms

        # Create figure
        fig = create_polymer_figure(
            all_coords, bonds, atom_colors, atoms, residue,
            selected_idx=selected_res
        )

        # Compute link distances
        o3p_col = atoms.index(residue.O3p.value)
        p_col = atoms.index(residue.P.value)

        link_dists = []
        for i in range(n_residues - 1):
            o3p = all_coords[i][o3p_col]
            p_next = all_coords[i + 1][p_col]
            dist = np.linalg.norm(p_next - o3p)
            link_dists.append(dist)

        # Compute geometry for selected residue
        sel_coords = all_coords[selected_res]
        geom = compute_geometry_stats(sel_coords, atoms, residue)

        # Build geometry display with color-coded deviations
        def format_geom(key, value, ref):
            if value is None:
                return None
            ref_mean = ref[key]['mean']
            ref_std = ref[key]['std']
            deviation = abs(value - ref_mean) / (ref_std + 1e-6)

            # Color code: green (<1σ), yellow (1-2σ), orange (2-3σ), red (>3σ)
            if deviation < 1:
                color = '#28a745'  # green
            elif deviation < 2:
                color = '#ffc107'  # yellow
            elif deviation < 3:
                color = '#fd7e14'  # orange
            else:
                color = '#dc3545'  # red

            unit = "°" if key.count("-") == 2 else "Å"
            return html.P(
                f"{key}: {value:.3f} {unit} ({deviation:.1f}σ)",
                style={'color': color, 'margin': '2px 0', 'fontSize': '11px'}
            )

        bond_items = []
        angle_items = []
        for key, value in geom.items():
            item = format_geom(key, value, ref_geometry)
            if item:
                if key.count("-") == 2:
                    angle_items.append(item)
                else:
                    bond_items.append(item)

        # Info panel
        info = html.Div([
            html.P(f"Selected: Residue {selected_res + 1}",
                   style={'fontWeight': 'bold', 'color': RESIDUE_COLORS[selected_res]}),
            html.P(f"Latent: [{', '.join(f'{v:.1f}' for v in slider_vals)}]",
                   style={'fontSize': '10px'}),
            html.Hr(),
            html.P("Bond Lengths:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            *bond_items,
            html.Hr(),
            html.P("Bond Angles:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            *angle_items,
            html.Hr(),
            html.P("O3'-P link distances:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            *[html.P(f"  Link {i+1}-{i+2}: {d:.2f}Å", style={'margin': '2px 0', 'fontSize': '11px'})
              for i, d in enumerate(link_dists)],
        ])

        new_store = {'latents': latents}

        return new_store, fig, info, *slider_vals

    print(f"\nStarting server at http://127.0.0.1:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
