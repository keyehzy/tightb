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

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from tightb.tightb import Graphene


class test_Graphene(unittest.TestCase):
    def test_periodic_inside_bounds(self):
        lattice = Graphene(2, 4)
        start_position = np.array([0, 0])
        delta = np.array([1, 0])

        end_position = lattice.periodic(start_position, delta)

        self.assertIsNone(assert_array_equal(end_position, np.array([1, 0])))

    def test_periodic_outside_bounds_in_x(self):
        lattice = Graphene(2, 4)
        start_position = np.array([0, 0])
        delta = np.array([2, 0])

        end_position = lattice.periodic(start_position, delta)

        self.assertIsNone(assert_array_equal(end_position, np.array([0, 0])))

    def test_periodic_outside_bounds_in_y(self):
        lattice = Graphene(2, 4)
        start_position = np.array([0, 0])
        delta = np.array([0, 4])

        end_position = lattice.periodic(start_position, delta)

        self.assertIsNone(assert_array_equal(end_position, np.array([0, 0])))

    def test_periodic_outside_bounds_in_x_y(self):
        lattice = Graphene(2, 4)
        start_position = np.array([0, 0])
        delta = np.array([2, 4])

        end_position = lattice.periodic(start_position, delta)

        self.assertIsNone(assert_array_equal(end_position, np.array([0, 0])))

    def test_periodic_with_orbitals_outside_bounds_in_x(self):
        lattice = Graphene(2, 4, orbitals=2)
        start_position = np.array([0, 0])
        delta = np.array([2, 0])

        end_position = lattice.periodic(start_position, delta)

        self.assertIsNone(assert_array_equal(end_position, np.array([0, 0])))

    def test_periodic_with_orbitals_outside_bounds_in_y(self):
        lattice = Graphene(2, 4, orbitals=2)
        start_position = np.array([0, 0])
        delta = np.array([0, 4])

        end_position = lattice.periodic(start_position, delta)

        self.assertIsNone(assert_array_equal(end_position, np.array([0, 0])))

    def test_periodic_with_orbitals_outside_bounds_in_x_y(self):
        lattice = Graphene(2, 4, orbitals=2)
        start_position = np.array([0, 0])
        delta = np.array([2, 4])

        end_position = lattice.periodic(start_position, delta)

        self.assertIsNone(assert_array_equal(end_position, np.array([0, 0])))

    def test_convert_coodinates(self):
        lattice = Graphene(2, 4)

        coordinate_in_grid_order = np.array([0, 0])
        coordinate_in_sequential_order = lattice.convert_coordinates(
                coordinate_in_grid_order)

        self.assertEqual(coordinate_in_sequential_order, 0)

        coordinate_in_grid_order = np.array([1, 0])
        coordinate_in_sequential_order = lattice.convert_coordinates(
                coordinate_in_grid_order)

        self.assertEqual(coordinate_in_sequential_order, lattice.dy)

        coordinate_in_grid_order = np.array([0, 1])
        coordinate_in_sequential_order = lattice.convert_coordinates(
                coordinate_in_grid_order)

        self.assertEqual(coordinate_in_sequential_order, 1)

    def test_convert_coodinates_with_orbitals(self):
        lattice = Graphene(2, 4, orbitals=2)

        coordinate_in_grid_order = np.array([0, 0])
        coordinate_in_sequential_order = lattice.convert_coordinates(
                coordinate_in_grid_order)

        self.assertEqual(coordinate_in_sequential_order, 0)

        coordinate_in_grid_order = np.array([1, 0])
        coordinate_in_sequential_order = lattice.convert_coordinates(
                coordinate_in_grid_order)

        self.assertEqual(coordinate_in_sequential_order,
                         lattice.orbitals * lattice.dy)

        coordinate_in_grid_order = np.array([0, 1])
        coordinate_in_sequential_order = lattice.convert_coordinates(
                coordinate_in_grid_order)

        self.assertEqual(coordinate_in_sequential_order, 1)

    def test_is_removed(self):
        lattice = Graphene(2, 4, sites_removed=[1, 2, 3, 4])
        self.assertTrue(lattice.is_removed([1, 2]))
        self.assertTrue(lattice.is_removed([2, 5]))
        self.assertFalse(lattice.is_removed([5, 6]))

    def test_check_removed(self):
        lattice = Graphene(2, 4, sites_removed=[1, 2, 3, 4])

        check = lattice.check_removed([1, 2], "foo")
        self.assertEqual(check, "0.0")

        check = lattice.check_removed([5, 6], "foo")
        self.assertEqual(check, "foo")
