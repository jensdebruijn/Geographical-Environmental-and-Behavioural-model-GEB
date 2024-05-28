# -*- coding: utf-8 -*-
import numpy as np
from honeybees.agents import AgentBaseClass


class NGO(AgentBaseClass):
    """This class is used to simulate NGOs.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["ngo"]
            if "ngo" in self.model.config["agent_settings"]
            else {}
        )
        AgentBaseClass.__init__(self)

    def step(self):
        """This function is run each timestep."""
        return None
