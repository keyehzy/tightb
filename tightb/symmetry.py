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


# Now the more complicated cases where the axis is tilted by some angle alpha
def reflect_point_by_tilted_axis(x0: np.array,
                                 v: np.array,
                                 r_star: list = [0.0, 0.0]):
    # Here we are using the parametric form of the line function
    #
    # x = x0 + v_x * t
    # y = x0 + v_y * t
    #
    # v is the unit vector in the direction of the tilted axis
    v = v / np.sqrt(v @ v)

    # alpha is the angle between the tilted axis and the y axis
    yhat = np.array([0.0, 1.0])
    alpha = np.arccos(yhat @ v)

    x0_rot = np.array([(x0[0] - r_star[0]) * np.cos(alpha) -
                       (x0[1] - r_star[1]) * np.sin(alpha),
                       (x0[0] - r_star[0]) * np.sin(alpha) +
                       (x0[1] - r_star[1]) * np.cos(alpha)])

    # Reflect
    x0_ref = np.array([-x0_rot[0], x0_rot[1]])

    # Then rotate and translate back
    return np.array([
        r_star[0] + x0_ref[0] * np.cos(alpha) + x0_ref[1] * np.sin(alpha),
        r_star[1] - x0_ref[0] * np.sin(alpha) + x0_ref[1] * np.cos(alpha)
    ])
