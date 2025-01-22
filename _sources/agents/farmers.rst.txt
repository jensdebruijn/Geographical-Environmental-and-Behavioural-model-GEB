Farmers
##########

Farmers are at the center of the GEB model, communicating directly to the hydrological model. First, the farmers are initialized using the previously created datafiles in the preprocessing scripts. Every timestep :meth:`agents.farmers.Farmers.step` method is called first, and farmers harvest and sow crops. When crops are harvested, their havest is `stored` in the agent class. Then, the hydrological model is run, calling :meth:`agents.farmers.Farmers.abstract_water` from the water demand module of CWatM. Here, the farmers can decide based on the availability of water, soil moisture content as well as their own characteristics whether to irrigate, from which sources and how much. These decisions are immediately communicated back to the hydrological model which continues the calculations for that timestep. Once the hydrological model is done, the cycle repeats and calling :meth:`agents.farmers.Farmers.step` again.

.. automodule:: geb.agents.farmers
    :members: