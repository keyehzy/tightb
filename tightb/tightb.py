# tightb generate tight-binding hamiltoniann crystalline structures
# Copyright (C) 2021  Matheus S. M. Sousa
#
# This file is part of tightb.
#
# tightb is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tightb is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tightb.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from tightb.writers import FortranWriter

class GrapheneSites:
    def __init__(self):
        pass

    class A:
        def __init__(self):
            pass

        @staticmethod
        def delta():
            return np.array([
                [0, 1],    # up right
                [-1, 1],    # down right
                [0, -1],    # left
            ])

        @staticmethod
        def direction():
            return [1, 2, 3]

        @staticmethod
        def rashba(spin):
            if spin:    # spin up
                return [
                    "0.5 * rsoc * (-1.0 + comp * sqrt(3.0))",
                    "0.5 * rsoc * (-1.0 - comp * sqrt(3.0))", "-rsoc"
                ]
            else:
                return [
                    "0.5 * rsoc * ( 1.0 + comp * sqrt(3.0))",
                    "0.5 * rsoc * ( 1.0 - comp * sqrt(3.0))", "rsoc"
                ]

    class B:
        def __init__(self):
            pass

        @staticmethod
        def delta():
            return np.array([
                [0, -1],    # down left
                [1, -1],    # up left
                [0, 1],    # right
            ])

        @staticmethod
        def direction():
            return [-1, -2, -3]

        @staticmethod
        def rashba(spin):
            if spin:
                return [
                    "0.5 * rsoc * ( 1.0 - comp * sqrt(3.0))",
                    "0.5 * rsoc * ( 1.0 + comp * sqrt(3.0))", "rsoc"
                ]
            else:
                return [
                    "0.5 * rsoc * (-1.0 - comp * sqrt(3.0))",
                    "0.5 * rsoc * (-1.0 + comp * sqrt(3.0))", "-rsoc"
                ]

    class C:
        def __init__(self):
            pass

        @staticmethod
        def delta():
            return np.array([
                [1, 1],    # up right
                [0, 1],    # down right
                [0, -1],    # left
            ])

        @staticmethod
        def direction():
            return [1, 2, 3]

        @staticmethod
        def rashba(spin):
            if spin:    # spin up
                return [
                    "0.5 * rsoc * (-1.0 + comp * sqrt(3.0))",
                    "0.5 * rsoc * (-1.0 - comp * sqrt(3.0))", "-rsoc"
                ]
            else:
                return [
                    "0.5 * rsoc * ( 1.0 + comp * sqrt(3.0))",
                    "0.5 * rsoc * ( 1.0 - comp * sqrt(3.0))", "rsoc"
                ]

    class D:
        def __init__(self):
            pass

        @staticmethod
        def delta():
            return np.array([
                [-1, -1],    # down left
                [0, -1],    # up left
                [0, 1],    # right
            ])

        @staticmethod
        def direction():
            return [-1, -2, -3]

        @staticmethod
        def rashba(spin):
            if spin:
                return [
                    "0.5 * rsoc * ( 1.0 - comp * sqrt(3.0))",
                    "0.5 * rsoc * ( 1.0 + comp * sqrt(3.0))", "rsoc"
                ]
            else:
                return [
                    "0.5 * rsoc * (-1.0 - comp * sqrt(3.0))",
                    "0.5 * rsoc * (-1.0 + comp * sqrt(3.0))", "-rsoc"
                ]


class Graphene:
    def __init__(self,
                 dx,
                 dy,
                 writer=None,
                 orbitals=1,
                 rashba_soc=False,
                 external_mag=False,
                 sites_removed=[]):
        self.dx = dx
        self.dy = dy
        self.writer = writer
        self.orbitals = orbitals
        self.rashba_soc = rashba_soc
        self.external_mag = external_mag
        self.sites_removed = convert_to_orbital(sites_removed, self.orbitals)
        self.Site = GrapheneSites()

    def external_mag_mat(self, i, j) -> str:
        mat = [["j_ex * cos(theta)", "j_ex * sin(theta) * exp(-comp * phi)"],
               ["j_ex * sin(theta) * exp(comp * phi)", "-j_ex * cos(theta)"]]
        return mat[i][j]

    def periodic(self, start_position: np.array, delta: np.array) -> np.array:
        return np.array([(start_position[0] + delta[0]) % self.dx,
                         (start_position[1] + self.orbitals * delta[1]) %
                         (self.orbitals * self.dy)])

    def convert_coordinates(self, coordinate_in_grid: np.array) -> int:
        return coordinate_in_grid[
            0] * self.dy * self.orbitals + coordinate_in_grid[1]

    def is_removed(self, pair_coords: list) -> bool:
        return any(coord in self.sites_removed for coord in pair_coords)

    def check_removed(self, pair_coords: list, result: str) -> str:
        return "0.0" if self.is_removed(pair_coords) else result

    def print_matrix_component_spinless(self, x0, x0_coordinate, Site) -> None:
        for delta, direction in zip(Site.delta(), Site.direction()):
            x = self.periodic(x0, delta)
            x_coordinate = self.convert_coordinates(x)

            pair_coords = [x0_coordinate + 1, x_coordinate + 1]
            result = self.check_removed(pair_coords, "t")
            self.writer.matrix_element("tij1",
                                       direction,
                                       *pair_coords,
                                       result=result)

    def print_matrix_component_rashba(self, x0, x0_coordinate, Site) -> None:
        spin = x0_coordinate % 2 == 0

        for delta, direction, rashba in zip(Site.delta(), Site.direction(),
                                            Site.rashba(spin)):
            x = self.periodic(x0, delta)
            x_coordinate = self.convert_coordinates(x)

            pair_coords = []

            if spin:
                pair_coords = [x0_coordinate + 1, x_coordinate + 2]
            else:
                pair_coords = [x0_coordinate + 1, x_coordinate]

            result = self.check_removed(pair_coords, rashba)
            self.writer.matrix_element("tij2",
                                       direction,
                                       *pair_coords,
                                       result=result)

    def print_matrix_component_external_mag(self, x0, x0_coordinate,
                                            Site) -> None:
        spin = x0_coordinate % 2 == 0

        result = ""
        pair_coords = [x0_coordinate + 1, x0_coordinate + 1]

        if spin:
            result = self.check_removed(pair_coords,
                                        self.external_mag_mat(0, 0))
        else:
            result = self.check_removed(pair_coords,
                                        self.external_mag_mat(1, 1))

        self.writer.matrix_element("tij3", 0, *pair_coords, result=result)

        if spin:
            pair_coords = [x0_coordinate + 1, x0_coordinate + 2]
            result = self.check_removed(pair_coords,
                                        self.external_mag_mat(0, 1))
        else:
            pair_coords = [x0_coordinate + 1, x0_coordinate]
            result = self.check_removed(pair_coords,
                                        self.external_mag_mat(1, 0))

        self.writer.matrix_element("tij3", 0, *pair_coords, result=result)

    def normalize_site(self, coord: int) -> int:
        return int(np.ceil(coord // self.orbitals))

    def project_out_sites(self) -> None:
        for site in self.sites_removed:
            self.writer.comment(f"Orbital {site}")
            self.writer.matrix_element("tij0", 0, site, site, result=1000000.0)
            self.writer.newline()

    def lattice(self, print_matrix_component_fn) -> None:
        for di in range(self.dx):
            for dj in range(self.orbitals * self.dy):
                x0 = np.array([di, dj])
                x0_coordinate = self.convert_coordinates(x0)
                site_kind = self.normalize_site(x0_coordinate) % 4
                self.writer.comment(f"Orbital {x0_coordinate + 1}")
                if site_kind == 0:
                    print_matrix_component_fn(x0, x0_coordinate, self.Site.A)
                elif site_kind == 1:
                    print_matrix_component_fn(x0, x0_coordinate, self.Site.B)
                elif site_kind == 2:
                    print_matrix_component_fn(x0, x0_coordinate, self.Site.C)
                elif site_kind == 3:
                    print_matrix_component_fn(x0, x0_coordinate, self.Site.D)
                self.writer.newline()

def convert_to_orbital(sites: list, orbitals: int) -> list:
    res = []
    for site in sites:
        for orb in range(orbitals):
            res.append(orbitals * (site - 1) + 1 + orb)
    return res


def argument_scaffolding(args) -> None:

    writer = FortranWriter()

    if (args.lattice == 'graphene'):

        lattice = Graphene(dx=args.dx,
                           dy=args.dy,
                           orbitals=args.orbitals,
                           rashba_soc=args.rashba_soc,
                           external_mag=args.external_mag,
                           sites_removed=args.remove_sites,
                           writer=writer)

        if args.remove_sites:
            writer.comment("Sites projected out")
            lattice.project_out_sites()

        writer.comment("Nearest neighbors interaction")
        lattice.lattice(lattice.print_matrix_component_spinless)

        if args.rashba_soc:
            writer.comment("Rashba Spin Orbit Interaction")
            lattice.lattice(lattice.print_matrix_component_rashba)

        if args.external_mag:
            writer.comment("Exchange Interaction")
            lattice.lattice(lattice.print_matrix_component_external_mag)

    print(writer, end='')
