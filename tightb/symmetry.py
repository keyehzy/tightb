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

eps = 1e-15


def graphene_lattice_real_coordinates(nx: int,
                                      ny: int,
                                      removed_sites: list = []) -> list:
    coordinates = []

    def append_valid(indice, element):
        if (indice not in removed_sites):
            coordinates.append(element)

    for j in range(ny):
        x0 = np.array([0.0, j * np.sqrt(3.0)])
        append_valid(4 * nx * j, np.array(x0 + [0., 0.]))
        for i in range(nx):
            indice = 4 * i + 4 * nx * j
            for idx in range(3):
                append_valid(indice + idx + 1, x0 + unitcell_sequence[idx])
                x0 += unitcell_sequence[idx]

            if i < nx - 1:
                append_valid(indice + 4, x0 - graphene_delta[2])

            x0 -= graphene_delta[2]

    return sorted(np.round(coordinates, 9), key=lambda x: (x[0], x[1]))


# Lets handle only the cases parallel to x or y axis
def reflect_point_by_vertical_axis(x0: np.array, x_star: float) -> np.array:
    return np.array([x_star - x0[0], x0[1]])


def reflect_point_by_horizontal_axis(x0: np.array, y_star: float) -> np.array:
    return np.array([x0[0], y_star - x0[1]])


class Boundary:
    def __init__(self, xmin=-1e16, xmax=1e16, ymin=-1e16, ymax=1e16):

        assert xmax > xmin or (xmax == xmin == 0.0)
        assert ymax > ymin or (ymax == ymin == 0.0)

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __str__(self):
        return f"xmin = {self.xmin}, xmax = {self.xmax}, ymin = {self.ymin}," \
                f"ymax =  {self.ymax}"

    def apply_boundary(self, x0: np.array) -> np.array:
        x, y = x0
        if x >= self.xmax:
            x -= (self.xmax - self.xmin)
        elif x < self.xmin:
            x += (self.xmax - self.xmin)

        if y >= self.ymax:
            y -= (self.ymax - self.ymin)
        elif y < self.ymin:
            y += (self.ymax - self.ymin)

        return np.array([x, y])


def boundary_from_lattice(lattice: list,
                          offset_x: float = 0.0,
                          offset_y: float = 0.0) -> Boundary:
    xmin = 0.0
    xmax = 0.0
    ymin = 0.0
    ymax = 0.0

    for coordinate in lattice:
        x, y = coordinate

        xmin = min(xmin, x)
        xmax = max(xmax, x)

        ymin = min(ymin, y)
        ymax = max(ymax, y)

    return Boundary(xmin - (offset_x + eps), xmax + (offset_x + eps),
                    ymin - (offset_y + eps), ymax + (offset_y + eps))


# Now the more complicated cases where the axis is tilted by some angle alpha
def reflect_point_by_tilted_axis(
        x0: np.array,
        v: np.array,
        r_star: list = [0.0, 0.0],
        boundary: Boundary = Boundary()) -> np.array:
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
    return boundary.apply_boundary(
            np.array([
                    r_star[0] + x0_ref[0] * np.cos(alpha) +
                    x0_ref[1] * np.sin(alpha), r_star[1] -
                    x0_ref[0] * np.sin(alpha) + x0_ref[1] * np.cos(alpha)
            ]))


def reflect_lattice_by_vertical_axis(lattice: list, x_star) -> list:
    reflected_lattice = []
    for site in lattice:
        reflected_lattice.append(reflect_point_by_vertical_axis(site, x_star))
    return reflected_lattice


def reflect_lattice_by_horizontal_axis(lattice: list, y_star) -> list:
    reflected_lattice = []
    for site in lattice:
        reflected_lattice.append(reflect_point_by_horizontal_axis(
                site, y_star))
    return reflected_lattice


def reflect_lattice_by_axis(
        lattice: list,
        v: np.array,
        r_star: list,
        boundary: Boundary = Boundary()) -> list:
    reflected_lattice = []
    for site in lattice:
        reflected_lattice.append(
                reflect_point_by_tilted_axis(site, v, r_star, boundary))
    return reflected_lattice


def is_symmetric_by_reflection(
        lattice: list,
        v: np.array,
        r_star: list,
        boundary: Boundary = Boundary()) -> bool:
    reflected_lattice = sorted(np.round(
            reflect_lattice_by_axis(lattice, v, r_star, boundary), 9),
                               key=lambda x: (x[0], x[1]))    # HACK(keyehz)
    return np.allclose(lattice, reflected_lattice)


def vertical_reflection_axis(lattice: list, boundary: Boundary) -> list:
    # TODO(keyezh): hardcoded, not sure if good enough but seems to pass all
    # the test so far
    N = 800
    dx = (boundary.xmax - boundary.xmin) / float(N)

    reflection_axes = []
    axis_direction = np.array([0.0, 1.0])    # parallel to y axis

    for j in range(N + 1):
        r = np.array([boundary.xmin + j * dx, 0.0])

        if is_symmetric_by_reflection(lattice, axis_direction, r, boundary):
            reflection_axes.append(r)

    return reflection_axes


def horizontal_reflection_axis(lattice: list, boundary: Boundary) -> list:
    # TODO(keyezh): hardcoded, not sure if good enough but seems to pass all
    # the test so far
    N = 800
    dy = (boundary.ymax - boundary.ymin) / float(N)

    reflection_axes = []
    axis_direction = np.array([1.0, 0.0])    # parallel to x axis

    for j in range(N + 1):
        r = np.array([0.0, boundary.ymin + j * dy])

        if is_symmetric_by_reflection(lattice, axis_direction, r, boundary):
            reflection_axes.append(r)

    return reflection_axes


def translate_point_in_direction_by_amount(
        x0: np.array,
        v: np.array,
        shift_amount: float,
        boundary: Boundary = Boundary()) -> np.array:
    return boundary.apply_boundary(x0 + shift_amount * v)


def translate_lattice_in_direction_by_amount(
        lattice: list,
        v: np.array,
        shift_amount: float,
        boundary: Boundary = Boundary()) -> list:
    translated_lattice = []
    for site in lattice:
        translated_lattice.append(
                translate_point_in_direction_by_amount(site, v, shift_amount,
                                                       boundary))
    return translated_lattice


def is_symmetric_by_translation(
        lattice: list,
        v: np.array,
        shift_amount: float,
        boundary: Boundary = Boundary()) -> bool:
    translated_lattice = sorted(np.round(
            translate_lattice_in_direction_by_amount(lattice, v, shift_amount,
                                                     boundary), 9),
                                key=lambda x: (x[0], x[1]))    # HACK(keyehz)
    return np.allclose(lattice, translated_lattice)


def perform_glide_operation_on_point_by_axis_by_amount(
        x0: np.array,
        v: np.array,
        r_star: list,
        shift_amount: float,
        boundary: Boundary = Boundary()) -> np.array:

    # Reflect by axis
    reflected_point = reflect_point_by_tilted_axis(x0, v, r_star, boundary)

    # Then shift
    translated_point = translate_point_in_direction_by_amount(
            reflected_point, v, shift_amount, boundary)

    # TODO(keyehzh): check if v is always the same (I think it is)

    return translated_point
