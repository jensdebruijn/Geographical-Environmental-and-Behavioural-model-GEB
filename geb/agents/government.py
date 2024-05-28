# -*- coding: utf-8 -*-
import numpy as np
from honeybees.agents import AgentBaseClass


class Government(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["government"]
            if "government" in self.model.config["agent_settings"]
            else {}
        )
        self.ratio_farmers_to_provide_subsidies_per_year = 0.05
        AgentBaseClass.__init__(self)

    def provide_subsidies(self) -> None:
        if not "subsidies" in self.config:
            return None
        if self.model.current_timestep == 1:
            for region in self.agents.farmers.borewell_cost_1[1].keys():
                self.agents.farmers.borewell_cost_1[1][region] = [
                    0.5 * x for x in self.agents.farmers.borewell_cost_1[1][region]
                ]
                self.agents.farmers.borewell_cost_2[1][region] = [
                    0.5 * x for x in self.agents.farmers.borewell_cost_2[1][region]
                ]

        return

    def request_flood_cushions(self, reservoirIDs):
        pass

    def set_irrigation_limit(self) -> None:
        if not "irrigation_limit" in self.config:
            return None
        irrigation_limit = self.config["irrigation_limit"]
        if irrigation_limit["per"] == "capita":
            self.agents.farmers.irrigation_limit_m3[:] = (
                self.agents.farmers.household_size * irrigation_limit["limit"]
            )
        else:
            raise NotImplementedError(
                "Only 'capita' is implemented for irrigation limit"
            )

    def step(self) -> None:
        """This function is run each timestep."""
        self.set_irrigation_limit()
        self.provide_subsidies()
