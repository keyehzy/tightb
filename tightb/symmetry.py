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

graphene_delta = np.array([[0.5, 0.5 * np.sqrt(3.0)],
                           [0.5, -0.5 * np.sqrt(3.0)], [-1.0, 0.0]])

unitcell_sequence = np.array(
    [graphene_delta[0], -graphene_delta[2], graphene_delta[1]])


def graphene_lattice_real_coordinates(nx: int, ny: int):
    coordinates = []
    for j in range(ny):
        x0 = np.array([0.0, j * np.sqrt(3.0)])
        coordinates.append(np.array(x0 + [0., 0.]))
        for i in range(nx):
            for idx in range(3):
                coordinates.append(x0 + unitcell_sequence[idx])
                x0 += unitcell_sequence[idx]

            if i < nx - 1:
                coordinates.append(x0 - graphene_delta[2])

            x0 -= graphene_delta[2]

    return coordinates


# Lets handle only the cases parallel to x or y axis
def reflect_point_by_vertical_axis(x0: np.array, x_star: float):
    return np.array([x_star - x0[0], x0[1]])


def reflect_point_by_horizontal_axis(x0: np.array, y_star: float):
    return np.array([x0[0], y_star - x0[1]])
