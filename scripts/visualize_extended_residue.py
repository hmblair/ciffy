#!/usr/bin/env python
"""
Interactive visualization of extended residue embedding with multiple residues.

Each residue is encoded with its coordinates PLUS the relative SE(3) transform
to the next residue. This allows the model to learn the coupling between
residue conformation and backbone geometry.

Features:
- Visualize 5 chained residues
- Dropdown to select which residue's PCs to modify
- Sliders control the selected residue's latent vector

Usage:
    python scripts/visualize_extended_residue.py
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
    position_next_residue,
)


# Atom colors by element
ATOM_COLORS = {
    'C': '#909090',  # gray
    'N': '#3050F8',  # blue
    'O': '#FF0D0D',  # red
    'P': '#FF8000',  # orange
    'H': '#FFFFFF',  # white
}

# Residue colors (for distinguishing residues)
RESIDUE_COLORS = [
    '#E41A1C',  # red
    '#377EB8',  # blue
    '#4DAF4A',  # green
    '#984EA3',  # purple
    '#FF7F00',  # orange
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
    try:
        for attr_name in dir(residue):
            attr = getattr(residue, attr_name)
            if hasattr(attr, 'value') and attr.value == atom_idx:
                name = attr_name
                if name.startswith('O'):
                    return 'O'
                elif name.startswith('N'):
                    return 'N'
                elif name.startswith('C'):
                    return 'C'
                elif name.startswith('P'):
                    return 'P'
                elif name.startswith('H'):
                    return 'H'
    except:
        pass
    return 'C'


def create_multi_residue_figure(
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
        # Determine color scheme
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
        title=dict(text=f'Residue Flow ({n_residues} Residues)', x=0.5),
    )

    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8052)
    parser.add_argument('--n-structures', type=int, default=100)
    parser.add_argument('--latent-dim', type=int, default=8)
    parser.add_argument('--n-residues', type=int, default=2)
    parser.add_argument('--n-epochs', type=int, default=100)
    args = parser.parse_args()

    n_residues = args.n_residues
    latent_dim = args.latent_dim

    print(f"Training ResidueFlowModel for adenosine...")
    data_dir = Path("/Users/hmblair/academic/data/rna-libraries/pdb/structures/pdb130")
    paths = sorted(data_dir.glob("*.cif"))[:args.n_structures]

    # Train the model using production class
    config = ResidueFlowConfig(
        latent_dim=latent_dim,
        n_layers=8,
        hidden_dim=64,
        bound=3.0,
        min_coverage=0.9,
    )

    model = ResidueFlowModel.from_structures(
        paths,
        Residue.A,
        config=config,
        n_epochs=args.n_epochs,
        verbose=True,
    )

    print(f"\nModel: {model}")

    # Get atom info
    atoms = model._atom_indices
    n_atoms = model.n_atoms
    residue = model.residue

    # Compute latent statistics from training data
    from ciffy.nn.residue_flow import extract_residues_with_links
    from ciffy.nn.residue_flow.data import compute_pca
    coords_all, transforms_all, _ = extract_residues_with_links(
        paths, Residue.A, min_coverage=0.9, verbose=False
    )
    coords_flat = coords_all.reshape(len(coords_all), -1)
    extended = np.concatenate([coords_flat, transforms_all], axis=1)
    X = torch.from_numpy(extended.astype(np.float32))

    with torch.no_grad():
        Z = model.flow.encode(X).numpy()
    z_mean = Z.mean(axis=0)
    z_std = Z.std(axis=0)

    print(f"\nLatent stats: mean~{z_mean.round(2)}, std~{z_std.round(2)}")

    # Reference coords (mean conformation in canonical frame)
    ref_coords = coords_all.mean(axis=0)

    # Bonds and colors
    bonds = get_residue_bonds(atoms, residue)
    atom_colors = [ATOM_COLORS.get(get_atom_element(a, residue), '#909090') for a in atoms]

    # Compute true variance per component from PCA singular values
    _, _, singular_values, var_explained = compute_pca(extended, n_components=latent_dim)
    total_var = (singular_values ** 2).sum()
    var_per_component = (singular_values ** 2) / total_var * 100
    print(f"PCA variance per component: {[f'{v:.1f}%' for v in var_per_component]}")

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
                marks={-3: '-3σ', 0: '0', 3: '+3σ'},
                tooltip={'placement': 'bottom'},
                updatemode='drag'
            ),
        ], style={'padding': '8px'}) for i in range(latent_dim)
    ]

    # Residue selector dropdown
    residue_options = [{'label': f'Residue {i+1}', 'value': i} for i in range(n_residues)]

    app.layout = html.Div([
        html.H2("Residue Flow - Multi-Residue Visualization",
                style={'textAlign': 'center'}),
        html.P(f"Model: {latent_dim}D latent, {model.var_explained*100:.1f}% var, "
               f"RMSD={model.pca_rmsd:.3f}Å | {n_residues} residues",
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
            # Load sliders from stored latent for newly selected residue
            slider_vals = latents[selected_res]
        else:
            # Update stored latent from sliders
            latents[selected_res] = slider_vals

        # Decode all residues
        all_coords = []
        all_transforms = []

        for res_idx in range(n_residues):
            z = z_mean + np.array(latents[res_idx]) * z_std
            z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)

            with torch.no_grad():
                coords, transform = model.decode(z_t)
                coords = coords.squeeze(0).numpy()
                transform = transform.squeeze(0).numpy()

            all_coords.append(coords)
            all_transforms.append(transform)

        # Position residues in chain
        positioned_coords = [all_coords[0]]  # First residue at origin

        for i in range(1, n_residues):
            prev_coords = positioned_coords[i - 1]
            curr_coords = all_coords[i]
            prev_transform = all_transforms[i - 1]

            # Position current residue relative to previous
            positioned = position_next_residue(
                prev_coords, curr_coords, prev_transform, atoms, residue
            )
            positioned_coords.append(positioned)

        # Create figure
        fig = create_multi_residue_figure(
            positioned_coords, bonds, atom_colors, atoms, residue,
            selected_idx=selected_res
        )

        # Compute link distances
        o3p_col = atoms.index(residue.O3p.value)
        p_col = atoms.index(residue.P.value)

        link_dists = []
        for i in range(n_residues - 1):
            o3p = positioned_coords[i][o3p_col]
            p_next = positioned_coords[i + 1][p_col]
            dist = np.linalg.norm(p_next - o3p)
            link_dists.append(dist)

        # Info panel
        sel_transform = all_transforms[selected_res]
        rot_angle = np.rad2deg(np.linalg.norm(sel_transform[:3]))
        trans_dist = np.linalg.norm(sel_transform[3:])

        info = html.Div([
            html.P(f"Selected: Residue {selected_res + 1}",
                   style={'fontWeight': 'bold', 'color': RESIDUE_COLORS[selected_res]}),
            html.P(f"Latent: [{', '.join(f'{v:.1f}' for v in slider_vals)}]"),
            html.P(f"Transform: rot={rot_angle:.1f}°, trans={trans_dist:.2f}Å"),
            html.Hr(),
            html.P("O3'-P link distances:", style={'fontWeight': 'bold'}),
            *[html.P(f"  Link {i+1}-{i+2}: {d:.2f}Å") for i, d in enumerate(link_dists)],
        ])

        new_store = {'latents': latents}

        return new_store, fig, info, *slider_vals

    print(f"\nStarting server at http://127.0.0.1:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
