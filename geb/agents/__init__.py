# -*- coding: utf-8 -*-
from .households import Households
from .farmers import Farmers
from .ngo import NGO
from .government import Government
from .reservoir_operators import ReservoirOperators


class Agents:
    """This class initalizes all agent classes, and is used to activate the agents each timestep.

    Args:
        model: The GEB model
    """

    def __init__(self, model) -> None:
        self.model = model
        self.households = Households(model, self, 0.1)
        self.farmers = Farmers(model, self, 0.1)
        self.reservoir_operators = ReservoirOperators(model, self)
        self._ngo = NGO(model, self)
        self.government = Government(model, self)

    def step(self) -> None:
        """This function is called every timestep and activates the agents in order of NGO, government and then farmers."""
        self.households.step()
        self._ngo.step()
        self.government.step()
        self.farmers.step()
        self.reservoir_operators.step()
