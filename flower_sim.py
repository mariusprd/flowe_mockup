# ------- IMPORTS ------- #
from collections import OrderedDict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp
from flwr.common import (
    Context,
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.server import Server as FlowerServer
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

import ray


from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


from typing import Tuple, Dict
Metrics = Dict[str, float]
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


from .my_client_proxy import CustomClientProxy
from .my_client_manager import SimpleClientManager




if __name__ == "__main__":
    NUM_CLIENTS = 10


    from .my_flower_client import Net, load_dataset, DEVICE
    def create_client(partition_id: int) -> Client:
        trainloader, valloader, _ = load_dataset(partition_id)
        return Client(
            partition_id=partition_id,
            net=Net().to(DEVICE),
            trainloader=trainloader,
            valloader=valloader
        )

    # Instantiate clients
    clients = [create_client(i) for i in range(NUM_CLIENTS)]


    # Instantiate some custom client proxies
    client_proxies = [CustomClientProxy(f"client_{i}", clients[i]) for i in range(NUM_CLIENTS)]

    my_client_manager = SimpleClientManager()
    for client_proxy in client_proxies:
        my_client_manager.register(client_proxy)



    # Set a basic strategy
    strategy = FedAvg(
       fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=10,  # Wait until all 10 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    )

    # Create my server
    my_server = FlowerServer(
        client_manager=my_client_manager,
        strategy=strategy
    )
    my_server.set_max_workers(10)