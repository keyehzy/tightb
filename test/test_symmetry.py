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
from numpy.testing import assert_allclose
import tightb.symmetry


class graphene_lattice_real_coordinates(unittest.TestCase):
    def test_onebyone(self):
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(1, 1)
        self.assertIsNone(
            assert_allclose(
                coordinates,
                np.array([[0.0, 0.0], [0.5, 0.8660254037844386],
                          [1.5, 0.8660254037844386], [2.0, 0.0]])))

    def test_twobytwo(self):
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(2, 2)
        self.assertIsNone(
            assert_allclose(
                coordinates,
                np.array([[0.0, 0.0], [0.0, 1.7320508075688772],
                          [0.5, 0.8660254037844386], [0.5, 2.598076211353316],
                          [1.5, 0.8660254037844386], [1.5, 2.598076211353316],
                          [2.0, 0.0], [2.0, 1.7320508075688774], [3.0, 0.0],
                          [3.0, 1.7320508075688774], [3.5, 0.8660254037844386],
                          [3.5, 2.598076211353316], [4.5, 0.8660254037844386],
                          [4.5, 2.598076211353316], [5.0, 0.0],
                          [5.0, 1.7320508075688774]])))


class reflect_point_by_vertical_axis(unittest.TestCase):
    def test_trivial(self):
        point = np.array([1.0, 1.0])
        y_axis = 0.0
        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_point_by_vertical_axis(point, y_axis),
                np.array([-1.0, 1.0])))

    def test_offaxis(self):
        point = np.array([1.0, 1.0])
        y_axis = 1.0
        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_point_by_vertical_axis(point, y_axis),
                np.array([0.0, 1.0])))


class reflect_point_by_horizontal_axis(unittest.TestCase):
    def test_trivial(self):
        point = np.array([1.0, 1.0])
        x_axis = 0.0
        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_point_by_horizontal_axis(
                    point, x_axis), np.array([1.0, -1.0])))

    def test_offaxis(self):
        point = np.array([1.0, 1.0])
        x_axis = 1.0
        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_point_by_horizontal_axis(
                    point, x_axis), np.array([1.0, 0.0])))


class reflect_point_by_tilted_axis(unittest.TestCase):
    def test_trivial(self):
        point = np.array([1.0, 1.0])
        n = np.array([0.0, 1.0])
        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_point_by_tilted_axis(point, n),
                np.array([-1.0, 1.0])))

    def test_tilted(self):
        point = np.array([1.0, 3.0])
        n = np.array([1.0, 1.0])    # 45 deg
        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_point_by_tilted_axis(point, n),
                np.array([3.0, 1.0])))    # Equal up to 1e-16

    def test_tilted_translation_y(self):
        point = np.array([3.0, 1.0])
        n = np.array([1.0, 1.0])    # 45 deg
        axis = [1.0, 0.0]
        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_point_by_tilted_axis(point, n, axis),
                np.array([2.0, 2.0])))    # Equal up to 1e-16

    def test_tilted_translation_x(self):
        point = np.array([3.0, 1.0])
        n = np.array([1.0, 1.0])    # 45 deg
        axis = [0.0, 1.0]
        self.assertIsNone(
            assert_allclose(tightb.symmetry.reflect_point_by_tilted_axis(
                point, n, axis),
                            np.array([0.0, 4.0]),
                            atol=1e-15))    # Equal up to 1e-15

    def test_tilted_translation_xy(self):
        point = np.array([3.0, 1.0])
        n = np.array([1.0, 1.0])    # 45 deg
        axis = [1.0, 1.0]
        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_point_by_tilted_axis(point, n, axis),
                np.array([1.0, 3.0])))    # Equal up to 1e-16


class reflect_lattice(unittest.TestCase):
    def test_vertical_all_on_x_axis(self):
        lattice = [
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([3.0, 0.0]),
            np.array([4.0, 0.0])
        ]

        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_lattice_by_vertical_axis(lattice, 0.0),
                [
                    np.array([-1.0, 0.0]),
                    np.array([-2.0, 0.0]),
                    np.array([-3.0, 0.0]),
                    np.array([-4.0, 0.0])
                ]))    # Equal up to 1e-16

    def test_vertical_all_on_y_axis(self):
        lattice = [
            np.array([0.0, 1.0]),
            np.array([0.0, 2.0]),
            np.array([0.0, 3.0]),
            np.array([0.0, 4.0])
        ]

        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_lattice_by_vertical_axis(lattice, 0.0),
                [
                    np.array([0.0, 1.0]),
                    np.array([0.0, 2.0]),
                    np.array([0.0, 3.0]),
                    np.array([0.0, 4.0])
                ]))    # Equal up to 1e-16

    def test_horizontal_all_on_x_axis(self):
        lattice = [
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([3.0, 0.0]),
            np.array([4.0, 0.0])
        ]

        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_lattice_by_horizontal_axis(
                    lattice, 0.0), [
                        np.array([1.0, 0.0]),
                        np.array([2.0, 0.0]),
                        np.array([3.0, 0.0]),
                        np.array([4.0, 0.0])
                    ]))    # Equal up to 1e-16

    def test_horizontal_all_on_y_axis(self):
        lattice = [
            np.array([0.0, 1.0]),
            np.array([0.0, 2.0]),
            np.array([0.0, 3.0]),
            np.array([0.0, 4.0])
        ]

        self.assertIsNone(
            assert_allclose(
                tightb.symmetry.reflect_lattice_by_horizontal_axis(
                    lattice, 0.0), [
                        np.array([0.0, -1.0]),
                        np.array([0.0, -2.0]),
                        np.array([0.0, -3.0]),
                        np.array([0.0, -4.0])
                    ]))    # Equal up to 1e-16


class is_symmetric_by_reflection_in_x_axis(unittest.TestCase):
    def test_trivial_true(self):
        lattice = [
            np.array([-2.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0])
        ]
        self.assertTrue(
            tightb.symmetry.is_symmetric_by_reflection(lattice,
                                                       np.array([0.0, 1.0]),
                                                       np.array([0.0, 0.0])))

    def test_trivial_false(self):
        lattice = [
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([3.0, 0.0]),
            np.array([4.0, 0.0])
        ]
        self.assertFalse(
            tightb.symmetry.is_symmetric_by_reflection(lattice,
                                                       np.array([0.0, 1.0]),
                                                       np.array([0.0, 0.0])))

    def test_offsite_true(self):
        lattice = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([3.0, 0.0]),
            np.array([4.0, 0.0])
        ]
        self.assertTrue(
            tightb.symmetry.is_symmetric_by_reflection(lattice,
                                                       np.array([0.0, 1.0]),
                                                       np.array([2.0, 0.0])))


class is_symmetric_by_reflection_in_y_axis(unittest.TestCase):
    def test_trivial_true(self):
        lattice = [
            np.array([0.0, -2.0]),
            np.array([0.0, -1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 2.0])
        ]
        self.assertTrue(
            tightb.symmetry.is_symmetric_by_reflection(lattice,
                                                       np.array([1.0, 0.0]),
                                                       np.array([0.0, 0.0])))

    def test_trivial_false(self):
        lattice = [
            np.array([0.0, 1.0]),
            np.array([0.0, 2.0]),
            np.array([0.0, 3.0]),
            np.array([0.0, 4.0])
        ]
        self.assertFalse(
            tightb.symmetry.is_symmetric_by_reflection(lattice,
                                                       np.array([1.0, 0.0]),
                                                       np.array([0.0, 0.0])))

    def test_offsite_true(self):
        lattice = [
            np.array([0.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 3.0]),
            np.array([0.0, 4.0])
        ]
        self.assertTrue(
            tightb.symmetry.is_symmetric_by_reflection(lattice,
                                                       np.array([1.0, 0.0]),
                                                       np.array([0.0, 2.0])))
