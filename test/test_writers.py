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
from tightb.writers import FortranWriter


class test_FortranWriter(unittest.TestCase):
    def test_simple(self):
        write = FortranWriter()
        write.append("hello world")
        self.assertEqual(write.toString(), "hello world\n")

    def test_append(self):
        write = FortranWriter()
        write.append("hello", "world")
        self.assertEqual(write.toString(), "hello world\n")

    def test_append_2(self):
        write = FortranWriter()
        write.append("hello")
        write.append("world")
        self.assertEqual(write.toString(), "hello\nworld\n")

    def test_comment(self):
        write = FortranWriter()
        write.comment("this is a comment")
        self.assertEqual(write.toString(), "! this is a comment\n")

    def test_matrix_element(self):
        write = FortranWriter()
        write.matrix_element("h", 1, 2, result="1.0")
        self.assertEqual(write.toString(), "h(1, 2) = 1.0\n")
