"""
Microgrid model for quantum optimization experiments.
Defines the 8-node test case based on modified IEEE 8-bus system.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import configparser
import json

@dataclass
class Generator:
    """Diesel generator model with quadratic cost function."""
    node: int
    min_power: float  # kW
    max_power: float  # kW
    cost_coeffs: Tuple[float, float, float]  # (a, b, c) for a*P² + b*P + c
    
    def production_cost(self, power: float) -> float:
        """Calculate production cost for given power output."""
        a, b, c = self.cost_coeffs
        return a * power**2 + b * power + c

@dataclass
class BatteryStorage:
    """Battery Energy Storage System (BESS) model."""
    node: int
    capacity: float  # kWh
    min_soc: float  # minimum state of charge (0-1)
    max_soc: float  # maximum state of charge (0-1)
    max_charge_power: float  # kW
    max_discharge_power: float  # kW
    efficiency: float  # round-trip efficiency
    degradation_cost: float  # $/kWh cycled
    current_soc: float = 0.6  # current state of charge
    
    def update_soc(self, power: float, time_interval: float = 1.0) -> float:
        """Update state of charge based on power flow."""
        if power > 0:  # discharging
            energy = power * time_interval / self.efficiency
        else:  # charging
            energy = power * time_interval * self.efficiency
        
        self.current_soc -= energy / self.capacity
        self.current_soc = max(self.min_soc, min(self.max_soc, self.current_soc))
        return self.current_soc

@dataclass
class RenewableSource:
    """Renewable energy source (PV) model."""
    node: int
    capacity: float  # kW
    curtailment_cost: float  # $/kWh curtailed
    current_generation: float = 0.0  # current available power

@dataclass
class Load:
    """Electrical load model."""
    node: int
    power: float  # kW
    is_critical: bool = True
    shedding_cost: float = 10.0  # $/kWh shed

class Microgrid:
    """Complete microgrid system model."""
    
    def __init__(self, config_file: str = "config/config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize microgrid components from configuration."""
        # Initialize generators
        self.generators = []
        gen_config = self.config["generators"]
        cost_coeffs = eval(gen_config["cost_coefficients"])
        limits = eval(gen_config["limits"])
        gen_nodes = [2, 5, 7]  # Node locations
        
        for i in range(len(cost_coeffs["a"])):
            self.generators.append(Generator(
                node=gen_nodes[i],
                min_power=limits["min"][i],
                max_power=limits["max"][i],
                cost_coeffs=(cost_coeffs["a"][i], cost_coeffs["b"][i], cost_coeffs["c"][i])
            ))
        
        # Initialize storage
        storage_config = self.config["storage"]
        self.storage = BatteryStorage(
            node=4,
            capacity=float(storage_config["capacity"]),
            min_soc=float(eval(storage_config["soc_limits"])[0]),
            max_soc=float(eval(storage_config["soc_limits"])[1]),
            max_charge_power=float(storage_config["max_power"]),
            max_discharge_power=float(storage_config["max_power"]),
            efficiency=float(storage_config["efficiency"]),
            degradation_cost=float(storage_config["degradation_cost"])
        )
        
        # Initialize renewable sources
        renewable_config = self.config["renewables"]
        capacities = eval(renewable_config["capacity"])
        curtailment_costs = eval(renewable_config["curtailment_cost"])
        renewable_nodes = [3, 6]
        
        self.renewables = []
        for i in range(len(capacities)):
            self.renewables.append(RenewableSource(
                node=renewable_nodes[i],
                capacity=capacities[i],
                curtailment_cost=curtailment_costs[i]
            ))
        
        # Initialize loads
        load_config = self.config["loads"]
        load_nodes = [1, 2, 5, 7, 8]  # Critical load locations
        load_powers = [400, 350, 450, 300, 300]  # kW
        
        self.loads = []
        for node, power in zip(load_nodes, load_powers):
            self.loads.append(Load(
                node=node,
                power=power,
                is_critical=True,
                shedding_cost=float(load_config["shedding_cost"])
            ))
    
    def set_operational_conditions(self, renewable_generation: List[float], load_demand: List[float]):
        """Set current renewable generation and load demand."""
        for i, ren in enumerate(self.renewables):
            ren.current_generation = renewable_generation[i]
        
        for i, load in enumerate(self.loads):
            load.power = load_demand[i]
    
    def power_balance_constraint(self, generator_powers: List[float], 
                                storage_power: float,
                                renewable_curtailment: List[float],
                                load_shedding: List[float]) -> float:
        """Calculate power balance constraint violation."""
        total_generation = sum(generator_powers) + storage_power
        total_renewable = sum(ren.current_generation for ren in self.renewables)
        total_curtailment = sum(renewable_curtailment)
        total_load = sum(load.power for load in self.loads)
        total_shedding = sum(load_shedding)
        
        return total_generation + (total_renewable - total_curtailment) - (total_load - total_shedding)
    
    def get_operational_cost(self, generator_powers: List[float],
                           storage_power: float,
                           renewable_curtailment: List[float],
                           load_shedding: List[float]) -> float:
        """Calculate total operational cost."""
        # Generator costs
        gen_cost = sum(gen.production_cost(power) 
                      for gen, power in zip(self.generators, generator_powers))
        
        # Storage degradation cost
        storage_cost = abs(storage_power) * self.storage.degradation_cost
        
        # Renewable curtailment cost
        curtailment_cost = sum(curt * ren.curtailment_cost 
                             for curt, ren in zip(renewable_curtailment, self.renewables))
        
        # Load shedding cost
        shedding_cost = sum(shed * load.shedding_cost 
                          for shed, load in zip(load_shedding, self.loads))
        
        return gen_cost + storage_cost + curtailment_cost + shedding_cost
    
    def check_constraints(self, generator_powers: List[float],
                         storage_power: float,
                         renewable_curtailment: List[float],
                         load_shedding: List[float]) -> Dict[str, bool]:
        """Check all operational constraints."""
        constraints = {}
        
        # Generator limits
        for i, (gen, power) in enumerate(zip(self.generators, generator_powers)):
            constraints[f"generator_{i}_limits"] = gen.min_power <= power <= gen.max_power
        
        # Storage power limits
        constraints["storage_power_limits"] = abs(storage_power) <= self.storage.max_discharge_power
        
        # Renewable curtailment limits
        for i, (ren, curt) in enumerate(zip(self.renewables, renewable_curtailment)):
            constraints[f"renewable_{i}_curtailment"] = 0 <= curt <= ren.current_generation
        
        # Load shedding limits
        for i, (load, shed) in enumerate(zip(self.loads, load_shedding)):
            constraints[f"load_{i}_shedding"] = 0 <= shed <= load.power
        
        # Power balance
        balance_violation = abs(self.power_balance_constraint(
            generator_powers, storage_power, renewable_curtailment, load_shedding))
        constraints["power_balance"] = balance_violation < 1e-3  # Tolerance
        
        return constraints

if __name__ == "__main__":
    # Test the microgrid model
    mg = Microgrid()
    print("Microgrid model initialized successfully!")
    print(f"Generators: {len(mg.generators)}")
    print(f"Renewables: {len(mg.renewables)}")
    print(f"Loads: {len(mg.loads)}")