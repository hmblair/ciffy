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
    parser.add_argument('--latent-dim', type=int, default=8)
    parser.add_argument('--n-residues', type=int, default=3)
    parser.add_argument('--n-epochs', type=int, default=100)
    args = parser.parse_args()

    n_residues = args.n_residues
    latent_dim = args.latent_dim

    print(f"Training ResidueFlowModel for adenosine...")
    data_dir = Path("/Users/hmblair/academic/data/rna-libraries/pdb/structures/pdb130")
    paths = sorted(data_dir.glob("*.cif"))[:args.n_structures]

    # Train the residue model
    config = ResidueFlowConfig(
        latent_dim=latent_dim,
        n_layers=8,
        hidden_dim=64,
        bound=3.0,
        min_coverage=0.9,
    )

    residue_model = ResidueFlowModel.from_structures(
        paths,
        Residue.A,
        config=config,
        n_epochs=args.n_epochs,
        verbose=True,
    )

    print(f"\nResidueFlowModel: {residue_model}")

    # Create PolymerFlowModel
    polymer_model = PolymerFlowModel({Residue.A: residue_model})
    print(f"PolymerFlowModel: {polymer_model}")

    # Get atom info
    atoms = residue_model._atom_indices
    n_atoms = residue_model.n_atoms
    residue = residue_model.residue

    # Compute latent statistics from training data
    coords_all, transforms_all, _ = extract_residues_with_links(
        paths, Residue.A, min_coverage=0.9, verbose=False
    )
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

    # Bonds and colors
    bonds = get_residue_bonds(atoms, residue)
    atom_colors = [ATOM_COLORS.get(get_atom_element(a, residue), '#909090') for a in atoms]

    # Create Dash app
    app = Dash(__name__)
    slider_range = 3.0

    # Sliders for latent dimensions
    sliders = [
        html.Div([
            html.Label(f"PC{i+1} ({var_per_component[i]:.1f}%)",
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
        State('latent-store', 'data'),
    )
    def update(selected_res, *args):
        from dash import ctx

        slider_vals = list(args[:latent_dim])
        reset_clicks = args[latent_dim]
        reset_all_clicks = args[latent_dim + 1]
        store_data = args[latent_dim + 2]

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

        # Build latent tensor: scale by z_std and add z_mean
        latent_array = np.array(latents)  # (n_residues, latent_dim)
        latent_scaled = z_mean + latent_array * z_std
        latent_tensor = torch.from_numpy(latent_scaled.astype(np.float32))

        # Decode using PolymerFlowModel
        sequence = [Residue.A] * n_residues
        with torch.no_grad():
            coords_flat = polymer_model.decode(latent_tensor, sequence)

        # Split back into per-residue coords for visualization
        all_coords = []
        offset = 0
        for _ in range(n_residues):
            coords_i = coords_flat[offset:offset + n_atoms].numpy()
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

        # Info panel
        info = html.Div([
            html.P(f"Selected: Residue {selected_res + 1}",
                   style={'fontWeight': 'bold', 'color': RESIDUE_COLORS[selected_res]}),
            html.P(f"Latent: [{', '.join(f'{v:.1f}' for v in slider_vals)}]"),
            html.Hr(),
            html.P("O3'-P link distances:", style={'fontWeight': 'bold'}),
            *[html.P(f"  Link {i+1}-{i+2}: {d:.2f}A") for i, d in enumerate(link_dists)],
        ])

        new_store = {'latents': latents}

        return new_store, fig, info, *slider_vals

    print(f"\nStarting server at http://127.0.0.1:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
