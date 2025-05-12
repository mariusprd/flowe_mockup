# Copyright 2022 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Flower ClientProxy."""


from typing import Optional

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
    Status,
)
from flwr.server.client_proxy import ClientProxy
from flwr.client import Client
import flwr

class CustomClientProxy(ClientProxy):
    def __init__(self, cid: str, client: Client):
        super().__init__(cid)
        self.client = client
        print(f"[Proxy {self.cid}] Created, linked to Client {client.partition_id}")

    def get_properties(self, ins: GetPropertiesIns, timeout: Optional[float], group_id) -> GetPropertiesRes:
        print(f"[Proxy {self.cid}] Delegating get_properties to Client {self.client.partition_id}  -> group_id: {group_id}")
        return self.client.get_properties(ins)

    def get_parameters(self, ins: GetParametersIns, timeout: Optional[float], group_id) -> GetParametersRes:
        print(f"[Proxy {self.cid}] Delegating get_parameters to Client {self.client.partition_id} -> group_id: {group_id}")
        return self.client.get_parameters(ins)

    def fit(self, ins: FitIns, timeout: Optional[float], group_id) -> FitRes:
        print(f"[Proxy {self.cid}] Delegating fit to Client {self.client.partition_id} -> group_id: {group_id}")
        return self.client.fit(ins)

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float], group_id) -> EvaluateRes:
        print(f"[Proxy {self.cid}] Delegating evaluate to Client {self.client.partition_id} -> group_id: {group_id}")
        return self.client.evaluate(ins)

    def reconnect(self, ins: flwr.common.ReconnectIns, timeout: Optional[float], group_id) -> flwr.common.DisconnectRes:
        # In this simulation, reconnect doesn't do much
        print(f"[Proxy {self.cid}] Reconnect called (no-op in simulation) -> group_id: {group_id}")
        return flwr.common.DisconnectRes(reason="Simulation reconnect (no-op)")

def test_cid() -> None:
    """Tests if the register method works correctly."""
    # Prepare
    cid_expected = "1"
    client_proxy = CustomClientProxy(cid=cid_expected)

    # Execute
    cid_actual = client_proxy.cid

    # Assert
    assert cid_actual == cid_expected


def test_properties_are_empty() -> None:
    """Tests if the register method works correctly."""
    # Prepare
    client_proxy = CustomClientProxy(cid="1")

    # Execute
    actual_properties = client_proxy.properties

    # Assert
    assert not actual_properties
