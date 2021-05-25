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
import tightb.symmetry


class graphene_lattice_real_coordinates(unittest.TestCase):
    def test_onebyone(self):
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(1, 1)
        self.assertIsNone(
            assert_array_equal(
                coordinates,
                np.array([[0.0, 0.0], [0.5, 0.8660254037844386],
                          [1.5, 0.8660254037844386], [2.0, 0.0]])))

    def test_twobytwo(self):
        coordinates = tightb.symmetry.graphene_lattice_real_coordinates(2, 2)
        self.assertIsNone(
            assert_array_equal(
                coordinates,
                np.array([[0.0, 0.0], [0.5, 0.8660254037844386],
                          [1.5, 0.8660254037844386], [2.0, 0.0], [3.0, 0.0],
                          [3.5, 0.8660254037844386], [4.5, 0.8660254037844386],
                          [5.0, 0.0], [0.0, 1.7320508075688772],
                          [0.5, 2.598076211353316], [1.5, 2.598076211353316],
                          [2.0, 1.7320508075688774], [3.0, 1.7320508075688774],
                          [3.5, 2.598076211353316], [4.5, 2.598076211353316],
                          [5.0, 1.7320508075688774]])))
