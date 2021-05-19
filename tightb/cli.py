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

import argparse
from tightb.tightb import argument_scaffolding


def scaffolding():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dx', type=int, required=True)
    parser.add_argument('--dy', type=int, required=True)
    parser.add_argument('--orbitals', type=int, default=1, choices=[1, 2])
    parser.add_argument('--rashba-soc', default=False, action='store_true')
    parser.add_argument('--external-mag', default=False, action='store_true')
    parser.add_argument('--remove-sites',
                        nargs='*',
                        type=int,
                        default=[],
                        required=False)
    parser.add_argument('--lattice', choices=['graphene'], required=True)
    args = parser.parse_args()

    if args.lattice == 'graphene':
        argument_scaffolding(args)
