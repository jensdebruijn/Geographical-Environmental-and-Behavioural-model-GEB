from numba.core.decorators import njit
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Optional


class DecisionModule:
    def __init__(self, agents) -> None:
        self.agents = agents

    @staticmethod
    @njit(cache=True)
    def IterateThroughFlood(
        n_floods: int,
        total_profits: np.ndarray,
        profits_no_event: np.ndarray,
        max_T: int,
        n_agents: int,
        discount_rate: np.ndarray,
    ) -> np.ndarray:
        """This function iterates through each (no)flood event i (see manuscript for details). It calculates the time discounted NPV of each event i:

        Args:
            n_floods: number of flood maps included in model run
            wealth: array containing the wealth of each household
            income: array containing the income of each household
            max_T: time horizon (same for each agent)
            expected_damages: array containing the expected damages of each household under all events i
            n_agents: number of household agents in the current admin unit
            r: time discounting rate

        Returns:
            NPV_summed: Array containing the summed time discounted NPV for each event i for each agent
        """

        # Allocate array
        NPV_summed = np.full((n_floods + 3, n_agents), -1, dtype=np.float32)

        # Iterate through all floods
        for i, index in enumerate(np.arange(1, n_floods + 3)):
            # Check if we are in the last iterations
            if i < n_floods:
                NPV_flood_i = total_profits[i]
                NPV_flood_i = (NPV_flood_i).astype(np.float32)

            # if in the last two iterations do not subtract damages (probs of no event)
            elif i >= n_floods:
                NPV_flood_i = profits_no_event
                NPV_flood_i = NPV_flood_i.astype(np.float32)

            # iterate over NPVs for each year in the time horizon and apply time
            # discounting
            NPV_t0 = (profits_no_event).astype(np.float32)  # no flooding in t=0

            # Calculate time discounted NPVs
            t_arr = np.arange(1, max_T, dtype=np.float32)

            discounts = 1 / (1 + discount_rate.reshape(-1, 1)) ** t_arr
            NPV_tx = np.sum(discounts, axis=1) * NPV_flood_i

            # Add NPV at t0 (which is not discounted)
            NPV_tx += NPV_t0

            # Store result
            NPV_summed[index] = NPV_tx

        # Store NPV at p=0 for bounds in integration
        NPV_summed[0] = NPV_summed[1]

        return NPV_summed

    def calcEU_do_nothing(
        self,
        n_agents: int,
        risk_perception: np.ndarray,
        total_profits: np.ndarray,
        profits_no_event: np.ndarray,
        p_droughts: np.ndarray,
        T: np.ndarray,
        discount_rate: float,
        sigma: float,
        subjective: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """This function calculates the time discounted subjective utility of not undertaking any action.

        Args:
            n_agents: number of agents in the floodplain.
            wealth: array containing the wealth of each household.
            income: array containing the income of each household.
            risk_perception: array containing the risk perception of each household (see manuscript for details).
            expected_damages: array expected damages for each flood event for each agent under no implementation of dry flood proofing.
            adapted: array containing the adaptation status of each agent (1 = adapted, 0 = not adapted).
            p_droughts: array containing the exceedance probabilities of each flood event included in the analysis.
            T: array containing the decision horizon of each agent.
            r: time discounting factor.
            sigma: risk aversion setting.

        Returns:
            EU_do_nothing_array: array containing the time discounted subjective utility of doing nothing for each agent.
        """

        # Ensure p floods is in increasing order
        indices = np.argsort(p_droughts)
        total_profits = total_profits[indices]
        p_droughts = np.sort(p_droughts)

        # Preallocate arrays
        n_floods, n_agents = total_profits.shape
        p_all_events = np.full((p_droughts.size + 3, n_agents), -1, dtype=np.float32)

        # calculate perceived risk
        perc_risk = p_droughts.repeat(n_agents).reshape(p_droughts.size, n_agents)

        # If
        if subjective == True:
            perc_risk *= risk_perception

        p_all_events[1:-2, :] = perc_risk

        # Cap percieved probability at 0.998. People cannot percieve any flood
        # event to occur more than once per year
        if np.max(p_all_events > 0.998):
            p_all_events[np.where(p_all_events > 0.998)] = 0.998

        # Add lasts p to complete x axis to 1 for trapezoid function (integrate
        # domain [0,1])
        p_all_events[-2, :] = p_all_events[-3, :] + 0.001
        p_all_events[-1, :] = 1

        # Add 0 to ensure we integrate [0, 1]
        p_all_events[0, :] = 0

        # Prepare arrays
        max_T = np.int32(np.max(T))

        # Part njit, iterate through floods
        n_agents = np.int32(n_agents)
        NPV_summed = self.IterateThroughFlood(
            n_floods,
            total_profits,
            profits_no_event,
            max_T,
            n_agents,
            discount_rate.data,
        )

        # Filter out negative NPVs
        NPV_summed = np.maximum(1, NPV_summed)

        if (NPV_summed == 1).any():
            print(f"Warning, {np.sum(NPV_summed == 1)} negative NPVs encountered")

        # Calculate expected utility
        ## NPV_Summed here is the wealth and income minus the expected damages of a certain probabilty event
        EU_store = (NPV_summed ** (1 - sigma)) / (1 - sigma)

        # Use composite trapezoidal rule integrate EU over event probability
        ## Here all
        y = EU_store
        x = p_all_events
        EU_do_nothing_array = np.trapz(y=y, x=x, axis=0)

        # People who already adapted cannot adapt, changed to a condition in the function that calls this, need the SEUT of doing nothing of those that have adapted
        # EU_do_nothing_array[np.where(adapted == 1)] = -np.inf

        return EU_do_nothing_array

    @staticmethod
    @njit(cache=True)
    def calcEU_adapt(
        expenditure_cap: float,
        loan_duration: int,
        n_agents: int,
        sigma: float,
        profits_no_event: np.ndarray,
        total_profits_adaptation: np.ndarray,
        profits_no_event_adaptation: np.ndarray,
        p_droughts: np.ndarray,
        risk_perception: np.ndarray,
        adaptation_costs: np.ndarray,
        total_annual_costs: np.ndarray,
        time_adapted: np.ndarray,
        adapted: np.ndarray,
        T: np.ndarray,
        discount_rate: float,
        extra_constraint: np.ndarray,
    ) -> np.ndarray:
        """This function calculates the discounted subjective utility for staying and implementing dry flood proofing measures for each agent.
        We take into account the current adaptation status of each agent, so that agents also consider the number of years of remaining loan payment.

        Args:
            expenditure_cap: expenditure cap for dry flood proofing investments.
            loan_duration: loan duration of the dry flood proofing investment.
            n_agents: number of agents present in the current floodplain.
            sigma: risk aversion setting of the agents.
            wealth: array containing the wealth of each household.
            income: array containing the income of each household.
            p_droughts: array containing the exceedance probabilities of each flood event included in the analysis.
            risk_perception: array containing the risk perception of each household.
            expected_damages_adapt: array of expected damages for each drought event if the farmer has adapted
            adaptation_costs: = array of annual implementation costs of the adaptation
            time_adapted: array containing the time each agent has been paying off their dry flood proofing investment loan.
            adapted: array containing the adaptation status of each agent (1 = adapted, 0 = not adapted).
            T: array containing the decision horizon of each agent.
            r: time discounting factor.

        Returns:
        EU_adapt: array containing the time discounted subjective utility of staying and implementing dry flood proofing for each agent.
        """

        # Preallocate arrays
        EU_adapt = np.full(n_agents, -1, dtype=np.float32)
        EU_adapt_dict = np.zeros(len(p_droughts) + 3, dtype=np.float32)

        t = np.arange(0, np.max(T), dtype=np.int32)

        # preallocate mem for flood array
        p_all_events = np.empty((p_droughts.size + 3), dtype=np.float32)

        # Ensure p floods is in increasing order
        indices = np.argsort(p_droughts)
        total_profits_adaptation = total_profits_adaptation[indices]
        p_droughts = np.sort(p_droughts)

        # Identify agents able to afford the adaptation and that have not yet adapted
        unconstrained = np.where(
            (profits_no_event * expenditure_cap > total_annual_costs)
            & (adapted == 0)
            & extra_constraint
        )
        # Create a mask to mask all constrained agents
        unconstrained_mask = (
            (profits_no_event * expenditure_cap > total_annual_costs)
            & (adapted == 0)
            & extra_constraint
        )

        # Those who cannot affort it cannot adapt
        EU_adapt[~unconstrained_mask] = -np.inf

        # Iterate only through agents who can afford to adapt
        for i in unconstrained[0]:
            # Find damages and loan duration left to pay
            total_profits_adaptation_i = total_profits_adaptation[:, i].copy()
            # expected_damages_adapt_i = expected_damages_adapt[:,i].copy()
            payment_remainder = max(loan_duration - time_adapted[i], 0)

            # Extract decision horizon
            t_agent = t[: T[i]]

            # NPV under no flood event
            NPV_adapt_no_flood = np.full(
                T[i], profits_no_event_adaptation[i], dtype=np.float32
            )

            NPV_adapt_no_flood[:payment_remainder] -= adaptation_costs[i]

            ## Calculate time discounted NPVs
            NPV_adapt_no_flood = np.sum(
                NPV_adapt_no_flood / (1 + discount_rate[i]) ** t_agent
            )

            # Apply utility function to NPVs
            EU_adapt_no_flood = (NPV_adapt_no_flood ** (1 - sigma[i])) / (1 - sigma[i])

            # # Calculate NPVs outcomes for each flood event
            NPV_adapt = np.empty((p_droughts.size, T[i]), dtype=np.float32)

            for j in range(p_droughts.size):
                NPV_adapt[j, :] = total_profits_adaptation_i[j]

            NPV_adapt[:, :payment_remainder] -= adaptation_costs[i]

            NPV_adapt /= (1 + discount_rate[i]) ** t_agent
            NPV_adapt_summed = np.sum(NPV_adapt, axis=1, dtype=np.float32)
            NPV_adapt_summed = np.maximum(
                np.full(NPV_adapt_summed.shape, 1, dtype=np.float32), NPV_adapt_summed
            )  # Filter out negative NPVs

            # if (NPV_adapt_summed == 1).any():
            #     print(
            #         f'Warning, {np.sum(NPV_adapt_summed == 1)} negative NPVs encountered')

            # Calculate expected utility
            EU_adapt_flood = (NPV_adapt_summed ** (1 - sigma[i])) / (1 - sigma[i])

            # Store results
            EU_adapt_dict[1 : EU_adapt_flood.size + 1] = EU_adapt_flood
            EU_adapt_dict[p_droughts.size + 1 : p_droughts.size + 3] = EU_adapt_no_flood
            EU_adapt_dict[0] = EU_adapt_flood[0]

            # Adjust for percieved risk
            p_all_events[1 : p_droughts.size + 1] = risk_perception[i] * p_droughts
            p_all_events[-2:] = p_all_events[-3] + 0.001, 1.0

            # Ensure we always integrate domain [0, 1]
            p_all_events[0] = 0

            # Integrate EU over probabilities trapezoidal
            EU_adapt[i] = np.trapz(EU_adapt_dict, p_all_events)

        return EU_adapt
