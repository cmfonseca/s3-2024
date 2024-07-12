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
import random
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
    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j
        logging.debug(f"LocalMove created with swap indices: {self.i}, {self.j}")

class Solution:
    def __init__(self, problem: Problem, order: Optional[List[int]] = None) -> None:
        self.problem = problem
        self.order = order if order is not None else self.random_permutation_sparse_fisher_yates_shuffle()
        self.objective_value = self.calculate_objective()
        self.best_solutions: List[Tuple[List[int], int, List[int], List[int]]] = []
        self.top_n = 5  # Number of top solutions to track

    def random_permutation_fisher_yates_shuffle(self) -> List[int]:
        """
        Generate a random permutation of the components using Fisher-Yates shuffle.
        """
        n = self.problem.n
        p = list(range(n))
        for i in range(n-1, 0, -1):
            j = random.randint(0, i)
            p[i], p[j] = p[j], p[i]
        logging.debug(f"Initial random permutation with Fisher-Yates shuffle: {p}")
        return p

    def random_permutation_sparse_fisher_yates_shuffle(self) -> List[int]:
        """
        Generate a random permutation of the components using Sparse Fisher-Yates shuffle.
        """
        n = self.problem.n
        p = dict()
        permutation = []
        for i in range(n-1, -1, -1):
            r = random.randint(0, i)
            permutation.append(p.get(r, r))
            if i != r:
                p[r] = p.get(i, i)
        permutation.reverse()  # Since we are appending, the order will be reversed
        logging.debug(f"Initial random permutation with Sparse Fisher-Yates shuffle: {permutation}")
        return permutation
    
    def calculate_objective_value(self, order: List[int]) -> Tuple[int, List[int], List[int]]:
        """Calculate the objective value, completion times, and individual costs for a given order."""
        if not order:
            return 0, [], []
        p, w, d = self.problem.p, self.problem.w, self.problem.d
        C = 0
        total_cost = 0
        individual_costs = []
        for i in order:
            C += p[i]
            T = max(C - d[i], 0)
            cost = w[i] * T
            individual_costs.append(cost)
            total_cost += cost
        logging.debug(f"Order: {order}, Completion times: {C}, Total cost: {total_cost}")
        return total_cost, [C] * len(order), individual_costs  # Adjust completion times if needed

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
        current_cost, _, current_individual_costs = self.calculate_objective_value(self.order)
        new_cost, _, new_individual_costs = self.calculate_objective_value(new_order)

        # Calculate the cost of already late tasks before and after adding the new component
        already_late_cost_before = sum(cost for cost in current_individual_costs if cost > 0)
        already_late_cost_after = sum(cost for cost in new_individual_costs if cost > 0)

        # Log late tasks with their cost
        late_tasks_before = [(i, cost) for i, cost in enumerate(current_individual_costs) if cost > 0]
        late_tasks_after = [(i, cost) for i, cost in enumerate(new_individual_costs) if cost > 0]
        
        logging.debug(f"Late tasks before adding component {component.cid}: {late_tasks_before}")
        logging.debug(f"Late tasks after adding component {component.cid}: {late_tasks_after}")

        # Calculate lookahead cost
        lookahead_cost = self.calculate_lookahead_cost(new_order, depth=2)

        # Increment includes new cost, the difference in already late costs, and the lookahead cost
        cost_incr = new_cost - current_cost + (already_late_cost_after - already_late_cost_before) + lookahead_cost
        logging.debug(f"Component {component.cid} would result in new cost: {new_cost}, increment: {cost_incr}, "
                    f"already late cost before: {already_late_cost_before}, already late cost after: {already_late_cost_after}, "
                    f"lookahead cost: {lookahead_cost}")
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
        return Solution(self.problem, self.order[:])

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

    def random_local_moves_wor(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves (in random order) that can be applied to
        the solution.
        """
        n = len(self.order)
        if n < 2:
            logging.debug("Order length less than 2, no local moves possible")
            return iter([])  # No local moves possible if order length is less than 2

        #indices = list(range(n))
        #random.shuffle(indices)
        # Use the random_permutation_sparse_fisher_yates_shuffle method to generate random indices
        random_indices = self.random_permutation_sparse_fisher_yates_shuffle()
        logging.debug(f"Generated random order of indices for local moves: {random_indices}")

        for i in range(n):
            for j in range(i + 1, n):
                logging.debug(f"Yielding LocalMove with swap indices: {random_indices[i]}, {random_indices[j]}")
                yield LocalMove(random_indices[i], random_indices[j])

    def objective_incr_local(self, lmove: LocalMove) -> Optional[int]:
        """
        Return the objective value increment resulting from applying a
        local move. If the objective value is not defined after
        applying the local move return None.
        """
        i, j = lmove.i, lmove.j
        if i >= len(self.order) or j >= len(self.order):
            logging.warning(f"LocalMove indices {i}, {j} are out of bounds for order length {len(self.order)}")
            return None
        
        # Swap the elements at indices i and j
        new_order = self.order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        # Calculate the new objective value
        current_cost = self.calculate_objective()
        new_cost, _, _ = self.calculate_objective_value(new_order)
        cost_incr = new_cost - current_cost
        
        logging.debug(f"Objective increment for LocalMove ({i}, {j}): {cost_incr}")
        return cost_incr
    
    def step(self, lmove: LocalMove) -> None:
        """
        Apply a local move to the solution.

        Note: this invalidates any previously generated components and
        local moves.
        """
        i, j = lmove.i, lmove.j
        if i >= len(self.order) or j >= len(self.order):
            logging.warning(f"LocalMove indices {i}, {j} are out of bounds for order length {len(self.order)}")
            return
        
        # Apply the swap
        logging.debug(f"Applying LocalMove with swap indices: {i}, {j}")
        self.order[i], self.order[j] = self.order[j], self.order[i]
        self.objective_value = self.calculate_objective()
        logging.debug(f"New order after LocalMove: {self.order}, New objective value: {self.objective_value}")

    def local_moves(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves that can be applied to the solution.
        """
        n = len(self.order)
        for i in range(n):
            for j in range(i + 1, n):
                yield LocalMove(i, j)


    def lower_bound(self) -> Optional[Objective]:
        """
        Return the lower bound value for this solution if defined,
        otherwise return None
        """
        # Weak lower bound: weighted tardiness so far
        lower_bound = self.calculate_objective()
#
        # Making it stronger could entail summing up the tardiness of the other elements
        # if they were to be run in parallel
        # Iterate through the indices of all object and consider those not yet in the order
        for j in range(self.problem.n):
            if j not in self.order:
                # Add the weighted tardiness of that element wrt the current processing time
                order_copy = self.order.copy()
                order_copy.append(j)
                C = 0
                for k in order_copy:
                    C += self.problem.p[k]
                T = max([C - self.problem.d[j], 0])
                lower_bound += self.problem.w[j] * T

        return lower_bound
    
    # same as above, but with different names
    #def lower_bound(self) -> Optional[Objective]:
    #    """
    #    Return the lower bound value for this solution if defined,
    #    otherwise return None
    #    """
#
    #    # Weak lower bound: weighted tardiness so far
    #    lower_bound = self.calculate_objective()
    #    current_completion_time = sum(self.problem.p[i] for i in self.order)
#
    #    # Iterate through the indices of all objects and consider those not yet in the order
    #    for j in range(self.problem.n):
    #        if j not in self.order:
    #            # Calculate completion time if this element were added next
    #            completion_time = current_completion_time + self.problem.p[j]
    #            # Calculate tardiness for this element
    #            tardiness = max(completion_time - self.problem.d[j], 0)
    #            # Update the lower bound with the weighted tardiness of the element
    #            lower_bound += self.problem.w[j] * tardiness
#
    #    return lower_bound
    
    def random_local_move(self) -> Optional[LocalMove]:
        """
        Return a random local move that can be applied to the solution.

        Note: repeated calls to this method may return the same
        local move.
        """
        if len(self.order) < 2:
            return None  # No local move possible if the order has fewer than 2 elements

        # Generate a random permutation of indices using the custom Fisher-Yates shuffle
        random_indices = self.random_permutation_sparse_fisher_yates_shuffle()

        # Select the first two indices from the random permutation
        i, j = random_indices[:2]

        return LocalMove(i, j)

    def perturb(self, ks: int) -> None:
        """
        Perturb the solution in place. The amount of perturbation is
        controlled by the parameter ks (kick strength)
        """
        for _ in range(ks):
            local_move = self.random_local_move()
            if local_move:
                self.step(local_move)

    def heuristic_add_move(self) -> Optional[Component]:
        """
        Return the next component to be added based on a heuristic rule.
        """
        # Choose one of the heuristics to use:

        # EDD: Earliest Due Date
        return self.heuristic_edd()

        # SPT: Shortest Processing Time
        #return self.heuristic_spt()

        # Priority: Smallest Weight
        #return self.heuristic_priority()

        # MST: Minimum Slack Time
        #return self.heuristic_mst()

        # WSPT: Weighted Shortest Processing Time
        #return self.heuristic_wspt()

    def heuristic_edd(self) -> Optional[Component]:
        logging.debug(f"Selecting component based on Earliest Due Date (EDD)")
        min_due_date = float('inf')
        selected_component = None

        for idx in range(self.problem.n):
            if idx not in self.order:
                due_date = self.problem.d[idx]
                if due_date < min_due_date:
                    min_due_date = due_date
                    selected_component = Component(idx)

        return selected_component

    def heuristic_spt(self) -> Optional[Component]:
        logging.debug(f"Selecting component based on Shortest Processing Time (SPT)")
        min_processing_time = float('inf')
        selected_component = None

        for idx in range(self.problem.n):
            if idx not in self.order:
                processing_time = self.problem.p[idx]
                if processing_time < min_processing_time:
                    min_processing_time = processing_time
                    selected_component = Component(idx)

        return selected_component

    def heuristic_priority(self) -> Optional[Component]:
        logging.debug(f"Selecting component based on Priority (Smallest Weight)")
        min_weight = float('inf')
        selected_component = None

        for idx in range(self.problem.n):
            if idx not in self.order:
                weight = self.problem.w[idx]
                if weight < min_weight:
                    min_weight = weight
                    selected_component = Component(idx)

        return selected_component

    def heuristic_mst(self) -> Optional[Component]:
        logging.debug(f"Selecting component based on Minimum Slack Time (MST)")
        min_slack_time = float('inf')
        selected_component = None

        for idx in range(self.problem.n):
            if idx not in self.order:
                slack_time = self.problem.d[idx] - self.problem.p[idx]
                if slack_time < min_slack_time:
                    min_slack_time = slack_time
                    selected_component = Component(idx)

        return selected_component

    def heuristic_wspt(self) -> Optional[Component]:
        logging.debug(f"Selecting component based on Weighted Shortest Processing Time (WSPT)")
        min_ratio = float('inf')
        selected_component = None

        for idx in range(self.problem.n):
            if idx not in self.order:
                ratio = self.problem.p[idx] / self.problem.w[idx]
                if ratio < min_ratio:
                    min_ratio = ratio
                    selected_component = Component(idx)

        return selected_component

    #bugged
    #def random_local_move(self) -> Optional[LocalMove]:
    #    """
    #    Return a random local move that can be applied to the solution.
#
    #    Note: repeated calls to this method may return the same
    #    local move.
    #    """
    #    i = random.randint(1, self.problem.n)
    #    j = i
    #    while j == i:
    #        j = random.randint(1, self.problem.n)
    #    raise LocalMove(i,j)

    #bugged
    #def perturb(self, ks: int) -> None:
    #    """
    #    Perturb the solution in place. The amount of perturbation is
    #    controlled by the parameter ks (kick strength)
    #    """
    #    for _ in range(ks):
    #        self.random_local_move()


class Problem:
    def __init__(self, processing_times: List[int], weights: List[int], due_dates: List[int]) -> None:
        self.p = processing_times
        self.w = weights
        self.d = due_dates
        self.n = len(processing_times)
        self.use_local_search = '--lsearch' in sys.argv and sys.argv[sys.argv.index('--lsearch') + 1] != 'none'

    @classmethod
    def from_textio(cls, f: TextIO) -> Problem:
        """
        Create a problem from a text I/O source `f`
        """
        nums = list(map(int, f.read().strip().split()))
        if len(nums) % 3 != 0:
            logging.error(f"Input length of numbers not divisible by 3")
            return None
        n = len(nums) // 3
        p = nums[:n]
        w = nums[n:2*n]
        d = nums[2*n:]

        print(p, w, d)
        return cls(p, w, d)
        
    def empty_solution(self) -> Solution:
        """
        Create a solution based on the presence of local search methods.
        """
        if self.use_local_search:
            # initial_order = list(range(self.n))
            # random.shuffle(initial_order)
            initial_order = list(Solution(self).random_permutation_sparse_fisher_yates_shuffle())
            logging.debug(f"Initial random permutation: {initial_order}")
            return Solution(self, initial_order)
        else:
            return Solution(self, [])

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
            if not p.use_local_search:
                logging.info("Best solutions found:")
                for order, cost, completion_times, individual_costs in s.best_solutions:
                    adjusted_order = [x + 1 for x in order]
                    logging.info(f"Component addition: Order: {adjusted_order}, Cost: {cost}, Completion times: {completion_times}")
                    logging.info(f"Cost calculation details: {[f'{adjusted_order[i]}: {individual_costs[i]}' for i in range(len(order))]}")
        else:
            logging.info(f"Objective: None")
    else:
        logging.info(f"Objective: no solution found")

    logging.info(f"Elapsed solving time: {end-start:.4f}")

