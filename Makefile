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

SHELL := /usr/bin/env bash
PYTHON ?= python3

tests = $(wildcard test/.py)

run:
	$(PYTHON) tightb.py

test:
	$(PYTHON) -m unittest -f $(tests)

.PHONY: run test
