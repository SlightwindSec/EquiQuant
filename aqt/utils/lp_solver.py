from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pulp


def get_constraints_for_search(
    max_weight_size: float,
    lower_bound: Optional[float] = None,
) -> Tuple[Dict[str, Tuple[float, float]]]:
    constraints = {
        "weight_size_after_compression": (
            lower_bound * max_weight_size if lower_bound else lower_bound,
            max_weight_size,
        )
    }
    return constraints, "weight_size_after_compression"


class LPS:
    """A wrapper on top of PuLP Linear Programming Solver.

    This solver maximizes/minimizes the candidate scores while meeting the cost constraints.
    """

    def __init__(
        self,
        name: str,
        # upper bound or (lower, upper) bound
        constraints: Dict[str, Union[float, Tuple[float, float]]],
        constraints_to_candidate_costs: Dict[str, List[List[float]]],
        candidate_scores: List[List[float]],
        objective_type: str = "minimize",
        verbose: bool = False,
    ) -> None:
        """Initialize the LPS solver."""
        assert set(constraints.keys()) == set(constraints_to_candidate_costs.keys())
        assert objective_type in ("minimize", "maximize")
        num_candidates_per_layer = list(map(len, candidate_scores))
        for candidate_costs in constraints_to_candidate_costs.values():
            assert list(map(len, candidate_costs)) == num_candidates_per_layer

        self.name = name
        self.constraints = constraints
        self.constraints_to_candidate_costs = constraints_to_candidate_costs
        self.candidate_scores = candidate_scores
        self.objective_type = (
            pulp.LpMinimize if objective_type == "minimize" else pulp.LpMaximize
        )
        self.solver = pulp.PULP_CBC_CMD(msg=verbose)

        self.num_layers = len(self.candidate_scores)
        self.num_candidates_per_layer = list(map(len, self.candidate_scores))

    def _build_selection_vars(self) -> list[list[pulp.LpVariable]]:
        vars = []
        for li in range(self.num_layers):
            num_candidates = self.num_candidates_per_layer[li]
            layer_vars = [
                pulp.LpVariable(f"z{li}_{ci}", lowBound=0, upBound=1, cat=pulp.LpBinary)
                for ci in range(num_candidates)
            ]
            vars.append(layer_vars)

        return vars

    def _build_objective_problem(
        self, selection_vars: List[List[pulp.LpVariable]]
    ) -> pulp.LpProblem:
        problem = pulp.LpProblem(name=self.name, sense=self.objective_type)
        objective_value = 0
        for layer_id, layer_vars in enumerate(selection_vars):
            objective_value += sum(
                [z * a for z, a in zip(layer_vars, self.candidate_scores[layer_id])]
            )

        problem += (objective_value, "L")
        return problem

    def _build_one_hot_constraints(
        self, selection_vars: List[List[pulp.LpVariable]]
    ) -> List[bool]:
        return [sum(layer_vars) == 1 for layer_vars in selection_vars]

    def _build_budget_constraints(
        self, selection_vars: List[List[pulp.LpVariable]]
    ) -> List[bool]:
        budget_constraints = []
        for (
            constraint_name,
            candidate_costs_list,
        ) in self.constraints_to_candidate_costs.items():
            cost = 0
            for layer_vars, candidate_costs in zip(
                selection_vars, candidate_costs_list
            ):
                cost += sum([z * b for z, b in zip(layer_vars, candidate_costs)])

            if isinstance(self.constraints[constraint_name], tuple):
                lower_bound, upper_bound = self.constraints[constraint_name]
            else:
                lower_bound, upper_bound = None, self.constraints[constraint_name]

            if upper_bound is not None:
                budget_constraints.append(cost <= upper_bound)

            if lower_bound is not None:
                budget_constraints.append(cost >= lower_bound)

        return budget_constraints

    def __call__(self) -> Tuple[List[int], str]:
        """Run the solver.

        Returns:
            selections: A list of selected candidate indices per layer.
            status: Status of the solver.
        """
        selection_vars = self._build_selection_vars()

        problem = self._build_objective_problem(selection_vars)
        for one_hot_constraint in self._build_one_hot_constraints(selection_vars):
            problem += one_hot_constraint

        for budget_constraint in self._build_budget_constraints(selection_vars):
            problem += budget_constraint

        problem.solve(self.solver)

        selections = [
            np.argmax([z.varValue for z in layer_vars]) for layer_vars in selection_vars
        ]
        return selections, pulp.LpStatus[problem.status]
