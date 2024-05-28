import numpy as np
import geopandas as gpd
import pyproj

from .general import AgentArray
from honeybees.agents import AgentBaseClass


class Households(AgentBaseClass):
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents

        locations = np.load(
            self.model.model_structure["binary"]["agents/households/locations"]
        )["data"]
        self.max_n = int(locations.shape[0] * (1 + reduncancy) + 1)

        self.locations = AgentArray(locations, max_n=self.max_n)

        sizes = np.load(
            self.model.model_structure["binary"]["agents/households/sizes"]
        )["data"]
        self.sizes = AgentArray(sizes, max_n=self.max_n)

        self.flood_depth = AgentArray(
            n=self.n, max_n=self.max_n, fill_value=False, dtype=bool
        )
        self.risk_perception = AgentArray(
            n=self.n, max_n=self.max_n, fill_value=1, dtype=np.float32
        )

        self.buildings = gpd.read_file(
            self.model.model_structure["geoms"]["assets/buildings"]
        )

        return None

    def flood(self, flood_map):
        self.flood_depth.fill(0)  # Reset flood depth for all households

        import matplotlib.pyplot as plt

        plt.figure()

        flood_map.plot()
        plt.savefig("flood.png")

        transformer = pyproj.Transformer.from_crs(
            4326, flood_map.raster.crs, always_xy=True
        )
        x, y = transformer.transform(self.locations[:, 0], self.locations[:, 1])

        forward_transform = flood_map.raster.transform
        backward_transform = ~forward_transform

        pixel_x, pixel_y = backward_transform * (x, y)
        pixel_x = pixel_x.astype(int)  # TODO: Should I add 0.5?
        pixel_y = pixel_y.astype(int)  # TODO: Should I add 0.5?

        # Create a mask that includes only the pixels inside the grid
        mask = (
            (pixel_x >= 0)
            & (pixel_x < flood_map.shape[1])
            & (pixel_y >= 0)
            & (pixel_y < flood_map.shape[0])
        )

        flood_depth_per_household = flood_map.values[pixel_y[mask], pixel_x[mask]]
        self.flood_depth[mask] = flood_depth_per_household > 0

        self.risk_perception[self.flood_depth] *= 10

        print("mean risk perception", self.risk_perception.mean())

        return None

    def step(self) -> None:
        self.risk_perception *= self.risk_perception
        return None

    @property
    def n(self):
        return self.locations.shape[0]
