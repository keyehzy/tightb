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

    def test_onebyone_remove_first(self):
        removed = [0]
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(
                1, 1, removed)
        self.assertIsNone(
                assert_allclose(
                        coordinates,
                        np.array([[0.5, 0.8660254037844386],
                                  [1.5, 0.8660254037844386], [2.0, 0.0]])))

    def test_onebyone_remove_second(self):
        removed = [1]
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(
                1, 1, removed)
        self.assertIsNone(
                assert_allclose(
                        coordinates,
                        np.array([[0.0, 0.0], [1.5, 0.8660254037844386],
                                  [2.0, 0.0]])))

    def test_onebyone_remove_third(self):
        removed = [2]
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(
                1, 1, removed)
        self.assertIsNone(
                assert_allclose(
                        coordinates,
                        np.array([[0.0, 0.0], [0.5, 0.8660254037844386],
                                  [2.0, 0.0]])))

    def test_onebyone_remove_forth(self):
        removed = [3]
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(
                1, 1, removed)
        self.assertIsNone(
                assert_allclose(
                        coordinates,
                        np.array([[0.0, 0.0], [0.5, 0.8660254037844386],
                                  [1.5, 0.8660254037844386]])))

    def test_twobyone(self):
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(2, 1)
        self.assertIsNone(
                assert_allclose(
                        coordinates,
                        np.array([[0.0, 0.0], [0.5, 0.8660254037844386],
                                  [1.5, 0.8660254037844386], [2.0, 0.0],
                                  [3.0, 0.0], [3.5, 0.8660254037844386],
                                  [4.5, 0.8660254037844386], [5.0, 0.0]])))

    def test_twobyone_remove_fifth(self):
        removed = [4]
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(
                2, 1, removed)
        self.assertIsNone(
                assert_allclose(
                        coordinates,
                        np.array([[0.0, 0.0], [0.5, 0.8660254037844386],
                                  [1.5, 0.8660254037844386], [2.0, 0.0],
                                  [3.5, 0.8660254037844386],
                                  [4.5, 0.8660254037844386], [5.0, 0.0]])))

    def test_onebytwo(self):
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(1, 2)
        self.assertIsNone(
                assert_allclose(
                        coordinates,
                        np.array([
                                [0.0, 0.0],
                                [0.0, 1.7320508075688772],
                                [0.5, 0.8660254037844386],
                                [0.5, 2.598076211353316],
                                [1.5, 0.8660254037844386],
                                [1.5, 2.598076211353316],
                                [2.0, 0.0],
                                [2.0, 1.7320508075688772],
                        ])))

    def test_onebytwo_remove_fifth(self):
        removed = [4]
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(
                1, 2, removed)
        self.assertIsNone(
                assert_allclose(
                        coordinates,
                        np.array([
                                [0.0, 0.0],
                                [0.5, 0.8660254037844386],
                                [0.5, 2.598076211353316],
                                [1.5, 0.8660254037844386],
                                [1.5, 2.598076211353316],
                                [2.0, 0.0],
                                [2.0, 1.7320508075688772],
                        ])))

    def test_twobytwo(self):
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(2, 2)
        self.assertIsNone(
                assert_allclose(
                        coordinates,
                        np.array([[0.0, 0.0], [0.0, 1.7320508075688772],
                                  [0.5, 0.8660254037844386],
                                  [0.5, 2.598076211353316],
                                  [1.5, 0.8660254037844386],
                                  [1.5, 2.598076211353316], [2.0, 0.0],
                                  [2.0, 1.7320508075688774], [3.0, 0.0],
                                  [3.0, 1.7320508075688774],
                                  [3.5, 0.8660254037844386],
                                  [3.5, 2.598076211353316],
                                  [4.5, 0.8660254037844386],
                                  [4.5, 2.598076211353316], [5.0, 0.0],
                                  [5.0, 1.7320508075688774]])))


class reflect_point_by_vertical_axis(unittest.TestCase):
    def test_trivial(self):
        point = np.array([1.0, 1.0])
        y_axis = 0.0
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_point_by_vertical_axis(
                                point, y_axis), np.array([-1.0, 1.0])))

    def test_offaxis(self):
        point = np.array([1.0, 1.0])
        y_axis = 1.0
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_point_by_vertical_axis(
                                point, y_axis), np.array([0.0, 1.0])))


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
        v = np.array([0.0, 1.0])
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_point_by_tilted_axis(x0=point,
                                                                     v=v),
                        np.array([-1.0, 1.0])))

    def test_tilted(self):
        point = np.array([1.0, 3.0])
        v = np.array([1.0, 1.0])    # 45 deg
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_point_by_tilted_axis(x0=point,
                                                                     v=v),
                        np.array([3.0, 1.0])))    # Equal up to 1e-16

    def test_tilted_translation_y(self):
        point = np.array([3.0, 1.0])
        v = np.array([1.0, 1.0])    # 45 deg
        axis = [1.0, 0.0]
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_point_by_tilted_axis(
                                x0=point, v=v, r_star=axis),
                        np.array([2.0, 2.0])))    # Equal up to 1e-16

    def test_tilted_translation_x(self):
        point = np.array([3.0, 1.0])
        v = np.array([1.0, 1.0])    # 45 deg
        axis = [0.0, 1.0]
        self.assertIsNone(
                assert_allclose(tightb.symmetry.reflect_point_by_tilted_axis(
                        x0=point, v=v, r_star=axis),
                                np.array([0.0, 4.0]),
                                atol=1e-15))    # Equal up to 1e-15

    def test_tilted_translation_xy(self):
        point = np.array([3.0, 1.0])
        v = np.array([1.0, 1.0])    # 45 deg
        axis = [1.0, 1.0]
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_point_by_tilted_axis(
                                x0=point, v=v, r_star=axis),
                        np.array([1.0, 3.0])))    # Equal up to 1e-16


class reflect_point_by_tilted_axis_with_boundary(unittest.TestCase):
    def test_trivial(self):
        point = np.array([2.0, 2.0])
        v = np.array([0.0, 1.0])
        boundary = tightb.symmetry.Boundary()
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_point_by_tilted_axis(
                                x0=point, v=v, boundary=boundary),
                        np.array([-2.0, 2.0])))

    def test_axis_in_y(self):
        point = np.array([2.0, 2.0])
        v = np.array([0.0, 1.0])
        boundary = tightb.symmetry.Boundary(xmin=-1.0, xmax=1.0)
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_point_by_tilted_axis(
                                x0=point, v=v, boundary=boundary),
                        np.array([0.0, 2.0])))

    def test_axis_in_x(self):
        point = np.array([2.0, 2.0])
        v = np.array([1.0, 0.0])
        boundary = tightb.symmetry.Boundary(ymin=-1.0, ymax=1.0)
        self.assertIsNone(
                assert_allclose(tightb.symmetry.reflect_point_by_tilted_axis(
                        x0=point, v=v, boundary=boundary),
                                np.array([2.0, 0.0]),
                                atol=1e-15))

    def test_axis_in_xy(self):
        point = np.array([2.0, 2.0])
        v = np.array([1.0, 1.0])
        boundary = tightb.symmetry.Boundary(xmin=-1.0,
                                            xmax=1.0,
                                            ymin=-1.0,
                                            ymax=1.0)
        self.assertIsNone(
                assert_allclose(tightb.symmetry.reflect_point_by_tilted_axis(
                        x0=point, v=v, boundary=boundary),
                                np.array([0.0, 0.0]),
                                atol=1e-15))

    def test_axis_in_x_assymetric(self):
        point = np.array([1.0, 1.0])
        v = np.array([0.0, 1.0])
        boundary = tightb.symmetry.Boundary(xmin=0.0, xmax=2.0)
        self.assertIsNone(
                assert_allclose(tightb.symmetry.reflect_point_by_tilted_axis(
                        x0=point, v=v, boundary=boundary),
                                np.array([1.0, 1.0]),
                                atol=1e-15))


class reflect_lattice(unittest.TestCase):
    def test_vertical_trivial(self):
        lattice = [
                np.array([0.0, 0.0]),
        ]
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_lattice_by_vertical_axis(
                                lattice, 0.0), [
                                        np.array([0.0, 0.0]),
                                ]))    # Equal up to 1e-16

    def test_vertical_all_on_x_axis(self):
        lattice = [
                np.array([1.0, 0.0]),
                np.array([2.0, 0.0]),
                np.array([3.0, 0.0]),
                np.array([4.0, 0.0])
        ]
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_lattice_by_vertical_axis(
                                lattice, 0.0), [
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
                        tightb.symmetry.reflect_lattice_by_vertical_axis(
                                lattice, 0.0), [
                                        np.array([0.0, 1.0]),
                                        np.array([0.0, 2.0]),
                                        np.array([0.0, 3.0]),
                                        np.array([0.0, 4.0])
                                ]))    # Equal up to 1e-16

    def test_vertical_all_off_axis(self):
        lattice = [
                np.array([1.0, 1.0]),
                np.array([2.0, 2.0]),
                np.array([3.0, 3.0]),
                np.array([4.0, 4.0])
        ]
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_lattice_by_vertical_axis(
                                lattice, 0.0), [
                                        np.array([-1.0, 1.0]),
                                        np.array([-2.0, 2.0]),
                                        np.array([-3.0, 3.0]),
                                        np.array([-4.0, 4.0])
                                ]))    # Equal up to 1e-16

    def test_horizontal_trivial(self):
        lattice = [
                np.array([0.0, 0.0]),
        ]
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_lattice_by_horizontal_axis(
                                lattice, 0.0), [
                                        np.array([0.0, 0.0]),
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

    def test_horizontal_off_axis(self):
        lattice = [
                np.array([1.0, 1.0]),
                np.array([2.0, 2.0]),
                np.array([3.0, 3.0]),
                np.array([4.0, 4.0])
        ]
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.reflect_lattice_by_horizontal_axis(
                                lattice, 0.0), [
                                        np.array([1.0, -1.0]),
                                        np.array([2.0, -2.0]),
                                        np.array([3.0, -3.0]),
                                        np.array([4.0, -4.0])
                                ]))    # Equal up to 1e-16


class is_symmetric_by_reflection_in_x_axis(unittest.TestCase):
    def test_trivial(self):
        lattice = [
                np.array([1.2345, 1.456789]),
                np.array([1.2345, 2.567890]),
                np.array([1.2345, 3.678901]),
                np.array([1.2345, 4.789012])
        ]
        self.assertTrue(
                tightb.symmetry.is_symmetric_by_reflection(
                        lattice, np.array([0.0, 1.0]), np.array([1.2345,
                                                                 0.0])))

    def test_trivial_true(self):
        lattice = [
                np.array([-2.0, 0.0]),
                np.array([-1.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([2.0, 0.0])
        ]
        self.assertTrue(
                tightb.symmetry.is_symmetric_by_reflection(
                        lattice, np.array([0.0, 1.0]), np.array([0.0, 0.0])))

    def test_trivial_false(self):
        lattice = [
                np.array([1.0, 0.0]),
                np.array([2.0, 0.0]),
                np.array([3.0, 0.0]),
                np.array([4.0, 0.0])
        ]
        self.assertFalse(
                tightb.symmetry.is_symmetric_by_reflection(
                        lattice, np.array([0.0, 1.0]), np.array([0.0, 0.0])))

    def test_offsite_true(self):
        lattice = [
                np.array([0.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([3.0, 0.0]),
                np.array([4.0, 0.0])
        ]
        self.assertTrue(
                tightb.symmetry.is_symmetric_by_reflection(
                        lattice, np.array([0.0, 1.0]), np.array([2.0, 0.0])))


class is_symmetric_by_reflection_in_y_axis(unittest.TestCase):
    def test_trivial(self):
        lattice = [
                np.array([1.456789, 1.2345]),
                np.array([2.567890, 1.2345]),
                np.array([3.678901, 1.2345]),
                np.array([4.789012, 1.2345])
        ]
        self.assertTrue(
                tightb.symmetry.is_symmetric_by_reflection(
                        lattice, np.array([1.0, 0.0]), np.array([0.0,
                                                                 1.2345])))

    def test_trivial_true(self):
        lattice = [
                np.array([0.0, -2.0]),
                np.array([0.0, -1.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, 2.0])
        ]
        self.assertTrue(
                tightb.symmetry.is_symmetric_by_reflection(
                        lattice, np.array([1.0, 0.0]), np.array([0.0, 0.0])))

    def test_trivial_false(self):
        lattice = [
                np.array([0.0, 1.0]),
                np.array([0.0, 2.0]),
                np.array([0.0, 3.0]),
                np.array([0.0, 4.0])
        ]
        self.assertFalse(
                tightb.symmetry.is_symmetric_by_reflection(
                        lattice, np.array([1.0, 0.0]), np.array([0.0, 0.0])))

    def test_offsite_true(self):
        lattice = [
                np.array([0.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, 3.0]),
                np.array([0.0, 4.0])
        ]
        self.assertTrue(
                tightb.symmetry.is_symmetric_by_reflection(
                        lattice, np.array([1.0, 0.0]), np.array([0.0, 2.0])))

    def test_graphene(self):
        lattice = tightb.symmetry.graphene_lattice_real_coordinates(1, 1)
        v = np.array([1.0, 0.0])
        r_star = np.array([0.0, 0.0])
        boundary = tightb.symmetry.boundary_from_lattice(
                lattice, 0.5, 0.5 * np.sqrt(3.0))
        self.assertFalse(
                tightb.symmetry.is_symmetric_by_reflection(
                        lattice, v, r_star, boundary))

    def test_graphene_glide_reflection(self):
        removed = [3, 4, 23, 24]
        lattice = tightb.symmetry.graphene_lattice_real_coordinates(
                4, 2, removed)
        boundary = tightb.symmetry.boundary_from_lattice(
                lattice, 0.25, 0.25 * np.sqrt(3.0))
        v = np.array([1.0, 0.0])
        r_star = np.array([0.0, np.sqrt(3.0)])
        self.assertTrue(
                tightb.symmetry.is_symmetric_by_reflection(
                        lattice, v, r_star, boundary))


class reflection_vertical_axis(unittest.TestCase):
    def test_centered_symmetric(self):
        lattice = [
                np.array([-2.0, 0.0]),
                np.array([-1.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([2.0, 0.0])
        ]
        boundary = tightb.symmetry.boundary_from_lattice(lattice)
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.vertical_reflection_axis(
                                lattice, boundary),
                        [[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]
                         ]))    # Equal up to 1e-16

    def test_centered_symmetric_float(self):
        lattice = [
                np.array([-2.123456789, 0.0]),
                np.array([-1.456789123, 0.0]),
                np.array([1.456789123, 0.0]),
                np.array([2.123456789, 0.0])
        ]
        boundary = tightb.symmetry.boundary_from_lattice(lattice)
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.vertical_reflection_axis(
                                lattice, boundary),
                        [[-2.123456789, 0.0], [0.0, 0.0], [2.123456789, 0.0]
                         ]))    # Equal up to 1e-16

    def test_centered_assymetric(self):
        lattice = [
                np.array([-2.0, 0.0]),
                np.array([-1.5, 0.0]),
                np.array([1.0, 0.0]),
                np.array([2.5, 0.0])
        ]
        boundary = tightb.symmetry.boundary_from_lattice(lattice)
        self.assertFalse(
                tightb.symmetry.vertical_reflection_axis(
                        lattice, boundary))    # Is empty

    def test_centered_assymetric_float(self):
        lattice = [
                np.array([-2.123456789, 0.0]),
                np.array([-1.567891234, 0.0]),
                np.array([1.0, 0.0]),
                np.array([2.567891234, 0.0])
        ]
        boundary = tightb.symmetry.boundary_from_lattice(lattice)
        self.assertFalse(
                tightb.symmetry.vertical_reflection_axis(
                        lattice, boundary))    # Is empty

    def test_graphene_simple(self):    # Graphene has three vertical axes
        lattice = tightb.symmetry.graphene_lattice_real_coordinates(1, 1)
        boundary = tightb.symmetry.boundary_from_lattice(
                lattice, 0.25, 0.25 * np.sqrt(3.0))
        self.assertIsNone(
                assert_allclose(tightb.symmetry.vertical_reflection_axis(
                        lattice, boundary),
                                [[-0.25, 0.0], [1.0, 0.0], [2.25, 0.0]],
                                atol=1e-15))

    def test_graphene_two_by_two(self):
        lattice = tightb.symmetry.graphene_lattice_real_coordinates(2, 2)
        boundary = tightb.symmetry.boundary_from_lattice(
                lattice, 0.25, 0.25 * np.sqrt(3.0))
        self.assertIsNone(
                assert_allclose(tightb.symmetry.vertical_reflection_axis(
                        lattice, boundary),
                                [[-0.25, 0.0], [2.5, 0.0], [5.25, 0.0]],
                                atol=1e-15))

    def test_graphene_glide_reflection(self):
        removed = [3, 4, 23, 24]
        lattice = tightb.symmetry.graphene_lattice_real_coordinates(
                4, 2, removed)
        boundary = tightb.symmetry.boundary_from_lattice(
                lattice, 0.25, 0.25 * np.sqrt(3.0))
        self.assertFalse(
                tightb.symmetry.vertical_reflection_axis(
                        lattice, boundary))    # Is empty

    def test_graphene_glide_glide(self):
        removed = [3, 15, 14, 18]
        lattice = tightb.symmetry.graphene_lattice_real_coordinates(
                2, 3, removed)
        boundary = tightb.symmetry.boundary_from_lattice(
                lattice, 0.25, 0.25 * np.sqrt(3.0))
        self.assertFalse(
                tightb.symmetry.vertical_reflection_axis(
                        lattice, boundary))    # Is empty


class reflection_horizontal_axis(unittest.TestCase):
    def test_centered_symmetric(self):
        lattice = [
                np.array([0.0, -2.0]),
                np.array([0.0, -1.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, 2.0])
        ]
        boundary = tightb.symmetry.boundary_from_lattice(lattice)
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.horizontal_reflection_axis(
                                lattice, boundary),
                        [[0.0, -2.0], [0.0, 0.0], [0.0, 2.0]
                         ]))    # Equal up to 1e-16

    def test_centered_symmetric_float(self):
        lattice = [
                np.array([0.0, -2.123456789]),
                np.array([0.0, -1.456789123]),
                np.array([0.0, 1.456789123]),
                np.array([0.0, 2.123456789])
        ]
        boundary = tightb.symmetry.boundary_from_lattice(lattice)
        self.assertIsNone(
                assert_allclose(
                        tightb.symmetry.horizontal_reflection_axis(
                                lattice, boundary),
                        [[0.0, -2.123456789], [0.0, 0.0], [0.0, 2.123456789]
                         ]))    # Equal up to 1e-16

    def test_centered_assymetric(self):
        lattice = [
                np.array([0.0, -2.0]),
                np.array([0.0, -1.5]),
                np.array([0.0, 1.0]),
                np.array([0.0, 2.5])
        ]
        boundary = tightb.symmetry.boundary_from_lattice(lattice)
        self.assertFalse(
                tightb.symmetry.horizontal_reflection_axis(
                        lattice, boundary))    # Is empty

    def test_centered_assymetric_float(self):
        lattice = [
                np.array([0.0, -2.123456789]),
                np.array([0.0, -1.567891234]),
                np.array([0.0, 1.0]),
                np.array([0.0, 2.567891234])
        ]
        boundary = tightb.symmetry.boundary_from_lattice(lattice)
        self.assertFalse(
                tightb.symmetry.horizontal_reflection_axis(
                        lattice, boundary))    # Is empty

    def test_graphene(self):
        lattice = tightb.symmetry.graphene_lattice_real_coordinates(1, 1)
        boundary = tightb.symmetry.boundary_from_lattice(
                lattice, 0.25, 0.25 * np.sqrt(3.0))
        self.assertIsNone(
                assert_allclose(tightb.symmetry.horizontal_reflection_axis(
                        lattice, boundary),
                                [[0.0, 0.0], [0.0, 0.5 * np.sqrt(3.0)]],
                                atol=1e-10))    # Equal up to 1e-10

    def test_graphene12(self):
        removed = [5, 6]
        lattice = tightb.symmetry.graphene_lattice_real_coordinates(
                1, 2, removed)
        boundary = tightb.symmetry.boundary_from_lattice(
                lattice, 0.25, 0.25 * np.sqrt(3.0))
        self.assertIsNone(
                assert_allclose(tightb.symmetry.horizontal_reflection_axis(
                        lattice, boundary), [[0.0, -0.25 * np.sqrt(3.0)],
                                             [0.0, 0.5 * np.sqrt(3.0)],
                                             [0.0, 1.25 * np.sqrt(3.0)]],
                                atol=1e-15))    # Equal up to 1e-16

    def test_graphene_glide_reflection(self):    # Wrong @@@
        removed = [3, 4, 23, 24]
        lattice = tightb.symmetry.graphene_lattice_real_coordinates(
                4, 2, removed)
        boundary = tightb.symmetry.boundary_from_lattice(
                lattice, 0.25, 0.25 * np.sqrt(3.0))
        self.assertIsNone(
                assert_allclose(tightb.symmetry.horizontal_reflection_axis(
                        lattice, boundary),
                                [[0.0, 0.0], [0.0, np.sqrt(3.0)]],
                                atol=1e-10))    # Equal up to 1e-16

    def test_graphene_glide_glide(self):
        removed = [3, 15, 14, 18]
        lattice = tightb.symmetry.graphene_lattice_real_coordinates(
                2, 3, removed)
        boundary = tightb.symmetry.boundary_from_lattice(
                lattice, 0.25, 0.25 * np.sqrt(3.0))
        self.assertFalse(
                tightb.symmetry.horizontal_reflection_axis(
                        lattice, boundary))    # Is empty
