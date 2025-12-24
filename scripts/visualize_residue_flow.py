#!/usr/bin/env python
"""
Interactive visualization of ResidueFlow latent space.

6 sliders control the 6 principal dimensions of adenosine conformations.
The 3D structure updates in real-time as you move the sliders.

Usage:
    python scripts/visualize_residue_flow.py

Requirements:
    pip install dash plotly

Author: Claude (Anthropic)
"""

import numpy as np
import torch
from pathlib import Path

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

from ciffy.biochemistry import Residue
from ciffy.nn.residue_flow.data import extract_residues, align_to_frame, compute_pca
from ciffy.nn.residue_flow.train import train_pca_flow


# Atom colors by element (inferred from atom type)
ATOM_COLORS = {
    'C': '#909090',  # gray
    'N': '#3050F8',  # blue
    'O': '#FF0D0D',  # red
    'P': '#FF8000',  # orange
    'H': '#FFFFFF',  # white
}

def get_residue_bonds(atoms: list, residue: Residue) -> list:
    """Get bond connectivity from residue topology, mapped to our atom subset."""
    # Map from residue atom index to our column index
    atom_to_col = {a: i for i, a in enumerate(atoms)}

    bonds = []
    for i, j in residue.bond_indices:
        if i in atom_to_col and j in atom_to_col:
            bonds.append((atom_to_col[i], atom_to_col[j]))

    return bonds


def get_atom_element(atom_idx: int, residue: Residue) -> str:
    """Get element symbol for an atom index."""
    # Get atom name from residue
    try:
        for attr_name in dir(residue):
            attr = getattr(residue, attr_name)
            if hasattr(attr, 'value') and attr.value == atom_idx:
                name = attr_name
                # First character is usually the element
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
    return 'C'  # default


def create_3d_figure(coords: np.ndarray, bonds: list, colors: list) -> go.Figure:
    """Create 3D molecular figure."""
    # Bond lines
    x_bonds, y_bonds, z_bonds = [], [], []
    for i, j in bonds:
        x_bonds.extend([coords[i, 0], coords[j, 0], None])
        y_bonds.extend([coords[i, 1], coords[j, 1], None])
        z_bonds.extend([coords[i, 2], coords[j, 2], None])

    fig = go.Figure()

    # Atoms
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(size=8, color=colors, opacity=0.9,
                   line=dict(width=1, color='black')),
        name='Atoms',
        hoverinfo='text',
        text=[f'Atom {i}' for i in range(len(coords))]
    ))

    # Bonds
    fig.add_trace(go.Scatter3d(
        x=x_bonds, y=y_bonds, z=z_bonds,
        mode='lines',
        line=dict(color='gray', width=4),
        name='Bonds',
        hoverinfo='skip'
    ))

    # Layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            bgcolor='white',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600,
        showlegend=False,
        title=dict(text='Adenosine Conformation', x=0.5)
    )

    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser(description='ResidueFlow Latent Space Visualizer')
    parser.add_argument('--port', type=int, default=8050, help='Port for Dash server')
    parser.add_argument('--n-structures', type=int, default=50, help='Number of structures for training')
    args = parser.parse_args()

    print("Loading adenosine data...")
    data_dir = Path("/Users/hmblair/academic/data/rna-libraries/pdb/structures/pdb130")
    paths = sorted(data_dir.glob("*.cif"))[:args.n_structures]

    coords_all, atoms = extract_residues(paths, Residue.A, min_coverage=0.9, verbose=True)
    coords_all = align_to_frame(coords_all, atoms, Residue.A)
    print(f"Loaded {len(coords_all)} adenosine instances with {len(atoms)} atoms")

    print("\nTraining 6D PCAFlow...")
    flow, info = train_pca_flow(
        coords_all,
        latent_dim=6,
        n_layers=8,
        hidden_dim=64,
        n_epochs=100,
        verbose=True,
    )
    flow.eval()

    # Get reference structure (mean)
    ref_coords = coords_all.mean(axis=0)

    # Get bonds from residue topology (fixed, not distance-based)
    bonds = get_residue_bonds(atoms, Residue.A)
    print(f"Bonds from topology: {len(bonds)}")

    # Get atom colors
    colors = [ATOM_COLORS.get(get_atom_element(a, Residue.A), '#909090') for a in atoms]

    # Compute latent space statistics for slider ranges
    with torch.no_grad():
        X = torch.from_numpy(coords_all).float()
        Z = flow.encode(X).numpy()

        # Encode mean structure - this is our "zero point" for sliders
        mean_coords_tensor = torch.from_numpy(ref_coords).float().unsqueeze(0)
        z_mean_struct = flow.encode(mean_coords_tensor).numpy()[0]

    z_std = Z.std(axis=0)
    z_mean = Z.mean(axis=0)

    print(f"\nLatent space statistics:")
    for i in range(6):
        print(f"  z[{i}]: mean={z_mean[i]:.2f}, std={z_std[i]:.2f}")
    print(f"\nMean structure encodes to: {z_mean_struct.round(2)}")

    # Variance explained by each component
    V, mean, singular_values, var_explained = compute_pca(coords_all, n_components=6)
    var_per_component = np.diff(np.concatenate([[0], var_explained])) * 100

    print(f"\nVariance per component: {var_per_component.round(1)}")

    # Create Dash app
    app = Dash(__name__)

    # Slider range matches model's bound
    slider_range = flow.bound if flow.bound else 3.0

    app.layout = html.Div([
        html.H2("Adenosine ResidueFlow Latent Space", style={'textAlign': 'center'}),
        html.P(f"Trained on {len(coords_all)} instances | RMSD: {info['pca_rmsd']:.3f}Å | Var: {info['var_explained']*100:.1f}%",
               style={'textAlign': 'center', 'color': 'gray'}),

        html.Div([
            # Left: 3D structure
            html.Div([
                dcc.Graph(id='structure-figure', style={'height': '600px'}),
            ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            # Right: Sliders
            html.Div([
                html.H4("Latent Dimensions", style={'textAlign': 'center'}),
                html.P("Move sliders to explore conformational space",
                       style={'textAlign': 'center', 'color': 'gray', 'fontSize': '12px'}),

                *[html.Div([
                    html.Label(f"PC{i+1} ({var_per_component[i]:.1f}% var)",
                              style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id=f'slider-{i}',
                        min=-slider_range,
                        max=slider_range,
                        step=0.1,
                        value=0,
                        marks={-3: '-3σ', 0: '0', 3: '+3σ'},
                        tooltip={'placement': 'bottom', 'always_visible': False},
                        updatemode='drag'  # Update while dragging
                    ),
                ], style={'padding': '15px'}) for i in range(6)],

                html.Div([
                    html.Button('Reset', id='reset-button', n_clicks=0,
                               style={'width': '100%', 'padding': '10px', 'marginTop': '20px'})
                ], style={'padding': '15px'}),

                html.Div(id='info-text', style={'padding': '15px', 'fontSize': '12px'})

            ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'padding': '20px', 'backgroundColor': '#f8f8f8'}),
        ]),
    ], style={'fontFamily': 'Arial, sans-serif'})

    @app.callback(
        Output('structure-figure', 'figure'),
        Output('info-text', 'children'),
        *[Output(f'slider-{i}', 'value') for i in range(6)],
        Input('slider-0', 'value'),
        Input('slider-1', 'value'),
        Input('slider-2', 'value'),
        Input('slider-3', 'value'),
        Input('slider-4', 'value'),
        Input('slider-5', 'value'),
        Input('reset-button', 'n_clicks'),
    )
    def update(z0, z1, z2, z3, z4, z5, reset_clicks):
        from dash import ctx

        # Reset if button clicked
        if ctx.triggered_id == 'reset-button':
            z_values = [0, 0, 0, 0, 0, 0]
        else:
            z_values = [z0, z1, z2, z3, z4, z5]

        # Slider values are offsets from mean structure in units of std dev
        z = z_mean_struct + np.array(z_values) * z_std

        # Decode to coordinates
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().unsqueeze(0)
            coords = flow.decode(z_tensor).squeeze(0).numpy()

        # Create figure (bonds are fixed from topology, not distance-based)
        fig = create_3d_figure(coords, bonds, colors)

        # Info text
        info = html.Div([
            html.P(f"Latent vector: [{', '.join(f'{v:.2f}' for v in z_values)}]"),
            html.P(f"Displacement from mean: {np.linalg.norm(coords - ref_coords):.3f}Å"),
        ])

        return fig, info, *z_values

    print(f"\nStarting server at http://127.0.0.1:{args.port}")
    print("Open this URL in your browser to interact with the visualization.")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
