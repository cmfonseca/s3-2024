#!/usr/bin/env python3
#
# Copyright (C) 2023 Alexandre Jesus <https://adbjesus.com>, Carlos M. Fonseca <cmfonsec@dei.uc.pt>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import TextIO, Optional, Any
from collections.abc import Iterable, Hashable

import logging

Objective = Any

class Component:
    model_index: int

    @property
    def cid(self) -> Hashable:
        # unique identifier for component used for hashing/comparison.
        return self.model_index

class LocalMove:

    # define a swap  
    i: int
    j: int

class Solution:

    def __init__(self, problem: Problem) -> None:
        self.cost = None  # Should we store the cost? should we calculate every time with self.objective?
        self.problem = problem
        self.u = [] # this is obviously temporary

    def output(self) -> str:
        """
        Generate the output string for this solution
        """
        return "\n".join(map(str, self.u))

    def copy(self) -> Solution:
        """
        Return a copy of this solution.

        Note: changes to the copy must not affect the original
        solution. However, this does not need to be a deepcopy.
        """
        # Check the TCP example. This is low-hanging fruit
        new_solution = Solution(self.problem)
        new_solution.u = self.u[:]  # copy of the sequence.
        new_solution.cost = self.cost  # also the cost
        return new_solution
        #raise NotImplementedError

    def is_feasible(self) -> bool:
        """
        Return whether the solution is feasible or not
        """
        # the length is 10, which is T (available timeslots)
        # and all u_i in u are smaller than M
        return len(self.u) == self.problem.T and all(
            map(lambda x: (0 <= x < self.problem.M), self.u))

    def objective(self) -> Optional[Objective]:
        """
        Return the objective value for this solution if defined, otherwise
        should return None
        """
        rp = [sum(self.problem.a[p][m] * self.problem.d[m] for m in range(self.problem.M)) \
              for p in range(self.problem.P)]

        cost = 0
        for t in range(1, self.problem.T + 1):
            for p in range(self.problem.P):
                actual_demand = sum(self.problem.a[p][self.u[i]] for i in range(t))
                target_demand = t * rp[p] / self.problem.T
                cost += (target_demand - actual_demand) ** 2
        return cost

    def lower_bound(self) -> Optional[Objective]:
        """
        Return the lower bound value for this solution if defined,
        otherwise return None
        """
        raise NotImplementedError

    def add_moves(self) -> Iterable[Component]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all components that can be added to the solution
        """
        if len(self.u) < self.problem.T:
            #generates all possible model indices possible to added.
            for model_index in range(self.problem.M):
                yield Component(model_index) #grabs one component as needed
        #raise NotImplementedError

    def local_moves(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves that can be applied to the solution
        """
        #all possible pairs of indices for swapping.
        for i in range(len(self.u)):
            for j in range(i + 1, len(self.u)):
                yield LocalMove(i, j) #grab one pair as needed
        #raise NotImplementedError

    def random_local_move(self) -> Optional[LocalMove]:
        """
        Return a random local move that can be applied to the solution.

        Note: repeated calls to this method may return the same
        local move.
        """
        if len(self.u) >= 2:
            i = random.randrange(len(self.u))
            j = random.randrange(len(self.u))
            return LocalMove(i, j)
        else:
            return None
        #raise NotImplementedError

    def random_local_moves_wor(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves (in random order) that can be applied to
        the solution.
        """
        moves = list(self.local_moves())
        random.shuffle(moves)
        return moves
        #raise NotImplementedError

    def heuristic_add_move(self) -> Optional[Component]:
        """
        Return the next component to be added based on some heuristic
        rule.
        """
        raise NotImplementedError

    def add(self, component: Component) -> None:
        """
        Add a component to the solution.

        Note: this invalidates any previously generated components and
        local moves.
        """
        self.u.append(component.model_index)
        self.cost = self.objective()  #update cost after adding component.
        #raise NotImplementedError

    def step(self, lmove: LocalMove) -> None:
        """
        Apply a local move to the solution.

        Note: this invalidates any previously generated components and
        local moves.
        """
        i, j = lmove.i, lmove.j
        self.u[i], self.u[j] = self.u[j], self.u[i]  #perform swap
        self.cost = self.objective()  #update cost after swap
        #raise NotImplementedError

    def objective_incr_local(self, lmove: LocalMove) -> Optional[Objective]:
        """
        Return the objective value increment resulting from applying a
        local move. If the objective value is not defined after
        applying the local move return None.
        """
        current_cost = self.objective()
        self.step(lmove)
        new_cost = self.objective()
        self.step(lmove)  #revert back to original state
        return new_cost - current_cost
        #raise NotImplementedError

    def lower_bound_incr_add(self, component: Component) -> Optional[Objective]:
        """
        Return the lower bound increment resulting from adding a
        component. If the lower bound is not defined after adding the
        component return None.
        """
        raise NotImplementedError

    def perturb(self, ks: int) -> None:
        """
        Perturb the solution in place. The amount of perturbation is
        controlled by the parameter ks (kick strength)
        """
        raise NotImplementedError

    def components(self) -> Iterable[Component]:
        """
        Returns an iterable to the components of a solution
        """
        return [Component(i) for i in range(self.problem.M)]




class Problem:
    @classmethod
    def from_textio(cls, f: TextIO) -> Problem:
        """
        Create a problem from a text I/O source `f`
        """
        data = f.readlines()
        M = int(data[0].split()[0])
        P = int(data[0].split()[1])
        d = [int(j) for j in data[1].split()]
        A = []
        for p in range(2,P):
            x = data[p].splitlines()[0].split()
            A.append([int(i) for i in x])

        return cls(M, P, d, A)

    def __init__(self, M, P, d, A) -> None:
        self.M = M
        self.P = P
        self.d = d
        self.A = A
        self.T = sum(d)

    def empty_solution(self) -> Solution:
        """
        Create an empty solution (i.e. with no components).
        """
        return None


if __name__ == '__main__':
    from api.solvers import *
    from time import perf_counter
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level',
                        choices=['critical', 'error', 'warning', 'info', 'debug'],
                        default='warning')
    parser.add_argument('--log-file', type=argparse.FileType('w'), default=sys.stderr)
    parser.add_argument('--csearch',
                        choices=['beam', 'grasp', 'greedy', 'heuristic', 'as', 'mmas', 'none'],
                        default='none')
    parser.add_argument('--cbudget', type=float, default=5.0)
    parser.add_argument('--lsearch',
                        choices=['bi', 'fi', 'ils', 'rls', 'sa', 'none'],
                        default='none')
    parser.add_argument('--lbudget', type=float, default=5.0)
    parser.add_argument('--input-file', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--output-file', type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()

    logging.basicConfig(stream=args.log_file,
                        level=args.log_level.upper(),
                        format="%(levelname)s;%(asctime)s;%(message)s")

    p = Problem.from_textio(args.input_file)
    s: Optional[Solution] = p.empty_solution()

    start = perf_counter()

    if s is not None:
        if args.csearch == 'heuristic':
            s = heuristic_construction(s)
        elif args.csearch == 'greedy':
            s = greedy_construction(s)
        elif args.csearch == 'beam':
            s = beam_search(s, 10)
        elif args.csearch == 'grasp':
            s = grasp(s, args.cbudget, alpha = 0.01)
        elif args.csearch == 'as':
            ants = [s]*100
            s = ant_system(ants, args.cbudget, beta = 5.0, rho = 0.5, tau0 = 1 / 3000.0)
        elif args.csearch == 'mmas':
            ants = [s]*100
            s = mmas(ants, args.cbudget, beta = 5.0, rho = 0.02, taumax = 1 / 3000.0, globalratio = 0.5)

    if s is not None:
        if args.lsearch == 'bi':
            s = best_improvement(s, args.lbudget)
        elif args.lsearch == 'fi':
            s = first_improvement(s, args.lbudget) 
        elif args.lsearch == 'ils':
            s = ils(s, args.lbudget)
        elif args.lsearch == 'rls':
            s = rls(s, args.lbudget)
        elif args.lsearch == 'sa':
            s = sa(s, args.lbudget, 30)

    end = perf_counter()

    if s is not None:
        print(s.output(), file=args.output_file)
        if s.objective() is not None:
            logging.info(f"Objective: {s.objective():.3f}")
        else:
            logging.info(f"Objective: None")
    else:
        logging.info(f"Objective: no solution found")

    logging.info(f"Elapsed solving time: {end-start:.4f}")

