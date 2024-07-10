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

from typing import TextIO, Optional, Any, List, Tuple
from collections.abc import Iterable, Hashable
import logging

Objective = Any

class Component:
    def __init__(self, index: int) -> None:
        self.index = index
        if self.index >= 0:
            logging.debug(f"Component created with index: {self.index}")

    @property
    def cid(self) -> Hashable:
        return self.index

class LocalMove:
    ...

class Solution:
    def __init__(self, problem: Problem, order: List[int], objective: int) -> None:
        self.problem = problem
        self.order = order
        self.objective_value = objective
        self.best_solutions: List[Tuple[List[int], int, List[int], List[int]]] = []
        self.top_n = 5  # Number of top solutions to track

    def calculate_objective_value(self, order: List[int]) -> Tuple[int, List[int], List[int]]:
        """Calculate the objective value, completion times, and individual costs for a given order."""
        if not order:
            return 0, [], []
        p, w, d = self.problem.p, self.problem.w, self.problem.d
        C = [0] * len(order)
        C[0] = p[order[0]]
        for i in range(1, len(order)):
            C[i] = C[i-1] + p[order[i]]
        individual_costs = [w[order[i]] * max(C[i] - d[order[i]], 0) for i in range(len(order))]
        total_cost = sum(individual_costs)
        logging.debug(f"Order: {order}, Completion times: {C}, Total cost: {total_cost}")
        return total_cost, C, individual_costs

    def calculate_objective(self) -> int:
        """Calculate the objective value based on the current order."""
        return self.calculate_objective_value(self.order)[0]
    
    def lower_bound_incr_add(self, component: Component) -> Optional[int]:
        """
        Return the lower bound increment resulting from adding a
        component. If the lower bound is not defined after adding the
        component return None.
        """
        new_order = self.order + [component.cid]
        current_cost = self.calculate_objective()
        new_cost, _, _ = self.calculate_objective_value(new_order)
        lookahead_cost = self.calculate_lookahead_cost(new_order, depth=4)  # Increased look-ahead depth
        cost_incr = new_cost - current_cost + lookahead_cost
        logging.debug(f"Component {component.cid} would result in new cost: {new_cost}, increment: {cost_incr}, lookahead: {lookahead_cost}")
        return cost_incr
    
    def calculate_lookahead_cost(self, order: List[int], depth: int = 1) -> int:
        """Calculate a lookahead cost for the next steps with specified depth."""
        if depth == 0:
            return 0

        remaining = set(range(self.problem.n)) - set(order)
        if not remaining:
            return 0

        min_future_cost = float('inf')

        for job in remaining:
            future_order = order + [job]
            future_cost, _, _ = self.calculate_objective_value(future_order)
            future_lookahead_cost = self.calculate_lookahead_cost(future_order, depth - 1)
            total_future_cost = future_cost + future_lookahead_cost

            if total_future_cost < min_future_cost:
                min_future_cost = total_future_cost

            if len(future_order) == self.problem.n:
                self.store_best_solution(future_order, total_future_cost)

        average_lookahead_cost = min_future_cost / len(remaining) if remaining else 0
        logging.debug(f"Lookahead cost: {average_lookahead_cost} for order: {order} with depth {depth}")
        return average_lookahead_cost

    def store_best_solution(self, order: List[int], cost: int):
        """Store the best solution in the top N list if applicable."""
        _, completion_times, individual_costs = self.calculate_objective_value(order)
        solution_tuple = (order, cost, completion_times, individual_costs)
        if solution_tuple not in self.best_solutions:
            self.best_solutions.append(solution_tuple)
            self.best_solutions.sort(key=lambda x: x[1])
            if len(self.best_solutions) > self.top_n:
                self.best_solutions.pop()
            logging.debug(f"Stored best solution: {order} with cost: {cost}")

    def output(self) -> str:
        """
        Generate the output string for this solution.
        The output should list the order in which jobs are processed, starting from 1.
        """
        return " ".join(map(str, [x + 1 for x in self.order]))

    def copy(self) -> 'Solution':
        """
        Return a copy of this solution.

        Note: changes to the copy must not affect the original
        solution. However, this does not need to be a deepcopy.
        """
        return Solution(self.problem, self.order[:], self.objective_value)

    def is_feasible(self) -> bool:
        """
        Return whether the solution is feasible or not
        """
        return len(self.order) == self.problem.n

    def objective(self) -> Optional[int]:
        """
        Return the objective value for this solution if defined, otherwise
        should return None
        """
        if not self.is_feasible():
            return None
        return self.objective_value
    
    def add(self, component: Component) -> None:
        """
        Add a component to the solution.

        Note: this invalidates any previously generated components and
        local moves.
        """
        logging.debug(f"Attempting to add component {component.cid} to order {self.order}")
        if component.cid >= self.problem.n:
            logging.warning(f"Component index {component.cid} is out of bounds")
            return
        self.order.append(component.cid)
        self.objective_value = self.calculate_objective()
        logging.debug(f"Added component {component.cid}, new objective: {self.objective_value}, new order: {self.order}")
        
        # Check and replace with the best solution found if needed
        if self.is_feasible() and self.best_solutions:
            best_order, best_cost, _, _ = self.best_solutions[0]
            if self.objective_value > best_cost:
                logging.info(f"Replacing final sequence with the best sequence found: {best_order} with cost: {best_cost}")
                self.order = best_order
                self.objective_value = best_cost

    def components(self) -> Iterable[Component]:
        """
        Returns an iterable to the components of a solution.
        """
        return [Component(cid) for cid in self.order]

    def add_moves(self) -> Iterable[Component]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all components that can be added to the solution
        """
        remaining = set(range(self.problem.n)) - set(self.order)
        for idx in remaining:
            yield Component(idx)

    def lower_bound(self) -> Optional[Objective]:
        """
        Return the lower bound value for this solution if defined,
        otherwise return None
        """
        raise NotImplementedError

    def local_moves(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves that can be applied to the solution
        """
        raise NotImplementedError

    def random_local_move(self) -> Optional[LocalMove]:
        """
        Return a random local move that can be applied to the solution.

        Note: repeated calls to this method may return the same
        local move.
        """
        raise NotImplementedError

    def random_local_moves_wor(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves (in random order) that can be applied to
        the solution.
        """
        raise NotImplementedError
            
    def heuristic_add_move(self) -> Optional[Component]:
        """
        Return the next component to be added based on some heuristic
        rule.
        """
        raise NotImplementedError

    def step(self, lmove: LocalMove) -> None:
        """
        Apply a local move to the solution.

        Note: this invalidates any previously generated components and
        local moves.
        """
        raise NotImplementedError

    def objective_incr_local(self, lmove: LocalMove) -> Optional[Objective]:
        """
        Return the objective value increment resulting from applying a
        local move. If the objective value is not defined after
        applying the local move return None.
        """
        raise NotImplementedError

    def perturb(self, ks: int) -> None:
        """
        Perturb the solution in place. The amount of perturbation is
        controlled by the parameter ks (kick strength)
        """
        raise NotImplementedError


class Problem:
    def __init__(self, processing_times: List[int], weights: List[int], due_dates: List[int]) -> None:
        self.p = processing_times
        self.w = weights
        self.d = due_dates
        self.n = len(processing_times)

    @classmethod
    def from_textio(cls, f: TextIO) -> Problem:
        """
        Create a problem from a text I/O source `f`
        """
        nums = list(map(int, f.read().strip().split()))
        n = len(nums) // 3
        p = nums[:n]
        w = nums[n:2*n]
        d = nums[2*n:]

        print(p, w, d)
        return cls(p, w, d)

    def empty_solution(self) -> Solution:
        """
        Create an empty solution (i.e. with no components).
        """
        return Solution(self, [], 0)


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
            logging.info("Best solutions found:")
            for order, cost, completion_times, individual_costs in s.best_solutions:
                adjusted_order = [x + 1 for x in order]
                logging.info(f"Order: {adjusted_order}, Cost: {cost}, Completion times: {completion_times}")
                logging.info(f"Cost calculation details: {[f'{adjusted_order[i]}: {individual_costs[i]}' for i in range(len(order))]}")
        else:
            logging.info(f"Objective: None")
    else:
        logging.info(f"Objective: no solution found")

    logging.info(f"Elapsed solving time: {end-start:.4f}")

