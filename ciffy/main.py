from __future__ import annotations
import numpy as np
import os
from enum import Enum
from .enum import Residue, RES_ABBREV, RibonucleicAcid, Element
from .reduction import Reduction, _Reduction, REDUCTIONS
from typing import Generator
import torch
from copy import copy
from ._c import _load


UNKNOWN = "UNKNOWN"


class Scale(Enum):
    RESIDUE  = 0
    CHAIN    = 1
    MOLECULE = 2


class Molecule(Enum):
    PROTEIN = 0
    RNA     = 1
    DNA     = 2
    WATER   = 3
    ION     = 4


def _molecule_type(val: int) -> Molecule:

    if val == 0:
        return Molecule.PROTEIN
    if val == 1:
        return Molecule.RNA
    if val == 2:
        return Molecule.DNA
    if val == 3:
        return Molecule.OTHER
    if val == 4:
        return Molecule.MISSING


def _all_equal(*values: int) -> bool:

    return len(set(values)) == 1


class Polymer:

    def __init__(
        self: Polymer,
        coordinates: torch.Tensor,
        atoms: torch.Tensor,
        elements: torch.Tensor,
        sequence: torch.Tensor,
        sizes: dict[Scale, torch.Tensor],
        id: str,
        names: list[str],
        lengths: torch.Tensor,
        nonpoly: int = 0,
    ) -> None:

        self._id = id
        self.names = names
        self.nonpoly = nonpoly

        if not _all_equal(
            coordinates.size(0),
            atoms.size(0),
            elements.size(0),
        ):
            raise ValueError(f"The coordinate, atom, and element tensors must be equal in size for PDB {self.id()}.")

        _res_count = sizes[Scale.RESIDUE].sum().item()
        _chn_count = sizes[Scale.CHAIN].sum().item()
        _mol_count = sizes[Scale.MOLECULE].sum().item()

        if not _all_equal(_res_count + nonpoly, _chn_count, _mol_count):
            raise ValueError(f"The residue ({_res_count} + {nonpoly}), chain ({_chn_count}), and molecule ({_mol_count}) atom counts do not agree for PDB {self.id()}.")

        self.coordinates = coordinates
        self.atoms = atoms
        self.elements = elements
        self.sequence = sequence
        self._sizes = sizes
        self.lengths = lengths

    def id(self: Polymer) -> str:
        """
        Get the PDB ID of the molecule.
        """

        if not self._id:
            return UNKNOWN
        return self._id

    def chain_id(self: Polymer, ix: int | None = None) -> str | list[str]:
        """
        Get the PDB + chain ID of each (or the specified) chain.
        """

        if ix is not None:
            return self.id() + '_' + self.names[ix]

        return [self.id() + '_' + name for name in self.names]

    def size(self: Polymer, scale: Scale | None = None) -> int:
        """
        Get the number of objects of the specified scale. Defaults to the
        number of atoms.
        """

        if scale is None:
            return self.coordinates.size(0)
        else:
            return self._sizes[scale].size(0)

    def type(self: Polymer) -> torch.Tensor:
        """
        Get the type of molecule (protein, RNA, DNA) of each chain.
        """

        types = torch.zeros(self.size(Scale.CHAIN), dtype=torch.long)
        atoms, _ = self.rreduce(self.sequence, Scale.CHAIN, Reduction.MAX)

        types[atoms < 5] = Molecule.RNA.value

        return types

    def reduce(
        self: Polymer,
        features: torch.Tensor,
        scale: Scale,
        rtype: Reduction = Reduction.MEAN,
    ) -> _Reduction:
        """
        Reduce the input values within each copy of the given object.
        MIN and MAX reductions return the indices too. A COLLATE
        reduction instead returns a list of tensors containing the
        values aligning with each specific index.
        """

        ATOM_DIM = 0

        count = self.size(scale)
        sizes = self._sizes[scale]
        ix = torch.arange(count).repeat_interleave(sizes)

        return REDUCTIONS[rtype](
            features, ix,
            dim=ATOM_DIM,
            dim_size=count,
        )

    def rreduce(
        self: Polymer,
        features: torch.Tensor,
        scale: Scale,
        rtype: Reduction = Reduction.MEAN,
    ) -> _Reduction:
        """
        Similar to reduce(), but with one feature per residue rather than per
        atom.
        """

        ATOM_DIM = 0

        count = self.size(scale)
        ix = torch.arange(count).repeat_interleave(self.lengths)

        return REDUCTIONS[rtype](
            features, ix,
            dim=ATOM_DIM,
            dim_size=count,
        )

    def expand(
        self: Polymer,
        features: torch.Tensor,
        scale: Scale,
    ) -> torch.Tensor:
        """
        Expand the features so that there is one per atom, rather than
        one per [scale].
        """

        return features.repeat_interleave(self._sizes[scale], dim=0)

    def center(
        self: Polymer,
        scale: Scale = Scale.MOLECULE,
    ) -> tuple[Polymer, torch.Tensor]:
        """
        Center the coordinates of each of the chosen object in the molecule.
        Return the centers of each [scale] well.
        """

        # Get the coordinate means

        means = self.reduce(self.coordinates, scale)

        # Expand the means and subtract

        expanded = self.expand(means, scale)
        coordinates = self.coordinates - expanded

        # Create a new Polymer with the centered coordinates

        centered = copy(self)
        centered.coordinates = coordinates

        return centered, means

    def _pc(
        self: Polymer,
        scale: Scale,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the principal components of all atoms for each instance
        of the given property.

        Since principal components are only defined up to sign,
        the outputs of this function may be unstable with respect to
        the coordinates. See the .align method for getting stable
        principal components.
        """

        # Get the coordinate covariance matrices at the given scale

        cov = self.coordinates[:, None, :] * self.coordinates[:, :, None]
        cov = self.reduce(cov, scale)

        # Compute the eigenvectors of the covariance matrices

        return torch.linalg.eigh(cov)

    def align(
        self: Polymer,
        scale: Scale,
    ) -> tuple[Polymer, torch.Tensor]:
        """
        Align the objects such that their covariance matrix is diagonal.
        Also return the principal components used to align the objects.
        """

        # Center and get the principal components

        aligned, _ = self.center(scale)
        _, Q = aligned._pc(scale)

        # Expand the components and use them to rotate the coordinates

        Q_exp = aligned.expand(Q, scale)
        aligned.coordinates = (
            Q_exp @ aligned.coordinates[..., None]
        ).squeeze()

        # To ensure stability and uniqueness, we modify the coordinates
        # and Q so that the largest two third moments are positive. The
        # third is chosen such that the system is right-handed.

        signs = aligned.moment(3, scale).sign()

        # The smallest eigenvalue is always the first

        signs[:, 0] = signs[:, 1] * signs[:, 2] * torch.linalg.det(Q)
        signs_exp = aligned.expand(signs, scale)

        # Modify the signs and return

        aligned.coordinates = aligned.coordinates * signs_exp
        Q = Q * signs[..., None]

        return aligned, Q

    def count(
        self: Polymer,
        mask: torch.Tensor,
        scale: Scale,
    ) -> torch.Tensor:
        """
        Count the nonzero elements per each [scale].
        """

        return self.reduce(mask.long(), scale, Reduction.SUM)

    def __getitem__(
        self: Polymer,
        mask: torch.Tensor,
    ) -> Polymer:
        """
        Return a polymer containing only the selected atoms.
        """

        coordinates = self.coordinates[mask]
        atoms = self.atoms[mask]
        elements = self.elements[mask]

        chain_sizes = self.count(mask, Scale.CHAIN)
        res_sizes = self.count(mask, Scale.RESIDUE)
        mol_sizes = self.count(mask, Scale.MOLECULE)

        _residues = torch.zeros(self.size(Scale.CHAIN), dtype=torch.bool)
        _residues[chain_sizes > 0] = True
        residues = _residues.repeat_interleave(self.lengths, dim=0)

        lengths = self.lengths[chain_sizes > 0]

        sizes = {
            Scale.RESIDUE: res_sizes[residues],
            Scale.CHAIN: chain_sizes[chain_sizes > 0],
            Scale.MOLECULE: mol_sizes,
        }

        sequence = self.sequence[residues]

        names = [
            self.names[ix] for ix in range(len(self.names))
            if chain_sizes[ix] > 0
        ]

        return Polymer(
            coordinates,
            atoms,
            elements,
            sequence,
            sizes,
            self._id,
            names,
            lengths,
        )

    def mask(
        self: Polymer,
        mask: torch.Tensor | int,
        scale: Scale,
    ) -> torch.Tensor:
        """
        Create a boolean mask to select only the indicated [scale]s.
        """

        counts = self.size(scale)
        objects = torch.zeros(counts, dtype=torch.bool)
        objects[mask] = True

        return self.expand(objects, scale)

    def select(
        self: Polymer,
        ix: torch.Tensor | int,
        scale: Scale,
    ) -> Polymer:
        """
        Create a new Polymer containing only the selected objects.
        """

        return self[self.mask(ix, scale)]

    def get_by_name(
        self: Polymer,
        name: torch.Tensor | int,
    ) -> Polymer:
        """
        Similar to select(), but for atom name rather than scale.
        """

        ix = (self.atoms[:, None] == name).any(1)
        return self[ix]

    def subset(
        self: Polymer,
        mol: Molecule,
    ) -> Polymer:
        """
        Similar to select(), but for molecule type rather than scale.
        """

        ix = (self.type() == mol.value).nonzero().squeeze(-1)
        return self.select(ix, Scale.CHAIN)

    def chains(
        self: Polymer,
        mol: Molecule | None = None
    ) -> Generator[Polymer]:
        """
        Yield chains from the polymer. Only yeild those of the given type if
        specified.
        """

        for ix in range(self.size(Scale.CHAIN)):
            chain = self.select(ix, Scale.CHAIN)
            if mol is None or chain.istype(mol):
                yield chain

    def missing(
        self: Polymer,
        scale: Scale = Scale.RESIDUE,
    ) -> torch.Tensor:
        """
        The indices of missing objects in the polymer. Defaults to missing
        residues.
        """

        return (self._sizes[scale] == 0).nonzero().squeeze(-1)

    def moment(
        self: Polymer,
        n: int,
        scale :Scale,
    ) -> torch.Tensor:
        """
        Return the n-th (uncentered) moment of the coordinates of the
        polymer at the given scale.
        """

        return self.reduce(self.coordinates ** n, scale)

    def __repr__(self: Polymer) -> str:

        out = ""

        out += f"PDB {self.id()} with {self.size()} atoms.\n"
        out += "───────────────────────────────────────\n"
        out += " " * len(f"{self.size(Scale.CHAIN)}") + "  "
        out += "Type     # Res  # Atom"
        out += "\n"

        types = self.type()

        for ix in range(self.size(Scale.CHAIN)):

            mol = _molecule_type(types[ix])

            _chain = f"{self.names[ix]}"
            _type = f"{mol.name}"
            _residues = f"{self.lengths[ix]}"
            _atoms = f"{self._sizes[Scale.CHAIN][ix]}"

            out += _chain + " " * (2 - len(_chain)) + "  "
            out += _type + " " * (9 - len(_type))
            out += _residues + " " * (7 - len(_residues))
            out += _atoms
            out += "\n"

        return out

    def str(self: Polymer) -> str:
        """
        Get the sequence of residues as a string.
        """

        DICT = lambda x: RES_ABBREV[Residue.revdict().get(x, 'N')]
        return "".join([DICT(ix.item()) for ix in self.sequence])

    def istype(
        self: Polymer,
        mol: Molecule,
    ) -> bool:
        """
        Check if the chain is the specified molecule type. Only works for
        single chains.
        """

        _type = self.type()
        if _type.size(0) != 1:
            return False
        return _type[0].item() == mol.value

    def write(self: Polymer, filename: str) -> None:

        with open(filename, 'w') as file:

            for chain in self.chains():

                atom = 0
                for residue in range(chain.size(Scale.RESIDUE)):

                    residue_name = self.str()[residue]

                    for _ in range(chain._sizes[Scale.RESIDUE][residue]):

                        element = Element.revdict()[chain.elements[atom].item()]
                        try:
                            atom_name = RibonucleicAcid.revdict()[chain.atoms[atom].item()].replace('p', '\'')
                        except Exception:
                            breakpoint()
                        file.write("ATOM  {:5d} {:4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           {:2s}\n".format(
                            atom + 1,
                            atom_name,
                            residue_name,
                            chain.names[0],
                            residue + 1,
                            chain.coordinates[atom][0],
                            chain.coordinates[atom][1],
                            chain.coordinates[atom][2],
                            element,
                        ))

                        atom += 1


def load(file: str) -> Polymer:

    if not os.path.isfile(file):
        raise OSError(f"The file \"{file}\" does not exist.")

    id, coordinates, atoms, elements, residues, atoms_per_res, atoms_per_chain, res_per_chain, chain_names, nonpoly = _load(file)

    mol_sizes = torch.tensor([len(coordinates)], dtype=torch.long).numpy()

    sizes = {
        Scale.RESIDUE: atoms_per_res,
        Scale.CHAIN: atoms_per_chain,
        Scale.MOLECULE: mol_sizes,
    }

    return Polymer(
        torch.from_numpy(coordinates),
        torch.from_numpy(atoms).long(),
        torch.from_numpy(elements).long(),
        torch.from_numpy(residues).long(),
        {key: torch.from_numpy(value).long() for key, value in sizes.items()},
        id,
        chain_names,
        torch.from_numpy(res_per_chain),
        nonpoly,
    )
