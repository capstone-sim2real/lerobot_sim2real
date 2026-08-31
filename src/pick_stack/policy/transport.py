"""Transport layer between the PICK client and the remote policy server.

``GrpcPolicyTransport`` speaks lerobot's async_inference protocol (the same
one the validated policy_server/robot_client pair uses); it imports lerobot,
grpc and torch lazily so pick_stack stays importable — and unit-testable —
without them. Tests substitute a fake transport.

Wire format notes (matching lerobot.async_inference):
  - observations go up as pickled TimedObservation
  - action chunks come down as pickled list[TimedAction]; each action tensor
    is converted here to a plain list[float] so the client core is torch-free
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any

from pick_stack.config import PolicyConfig

logger = logging.getLogger(__name__)

# (timestep, action vector) — vector order follows the robot's action_features
ActionStep = tuple[int, list[float]]


class PolicyTransport(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> None:
        """Handshake + send policy instructions (server loads the model here)."""

    @abc.abstractmethod
    def ping(self) -> bool:
        """Cheap health check; used by the FSM before entering PICK."""

    @abc.abstractmethod
    def send_observation(self, raw_observation: dict[str, Any], timestep: int, must_go: bool) -> bool: ...

    @abc.abstractmethod
    def poll_actions(self) -> list[ActionStep]:
        """Non-blocking-ish: whatever action chunk the server has ready, else []."""

    @abc.abstractmethod
    def close(self) -> None: ...


class GrpcPolicyTransport(PolicyTransport):
    """lerobot async_inference gRPC client bits, robot instance injected.

    ``lerobot_robot`` is the underlying lerobot Robot (So101RobotIO.robot) —
    only used to derive the observation/action feature spec the server needs.
    """

    def __init__(self, lerobot_robot, cfg: PolicyConfig):
        self._robot = lerobot_robot
        self._cfg = cfg
        self._channel = None
        self._stub = None

    def connect(self) -> None:
        import pickle  # nosec - trusted policy server, lerobot protocol

        import grpc

        from lerobot.async_inference.helpers import RemotePolicyConfig, map_robot_keys_to_lerobot_features
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options

        environment_dt = 1.0 / self._cfg.fps if self._cfg.fps > 0 else 0.033
        self._channel = grpc.insecure_channel(
            self._cfg.server_address, grpc_channel_options(initial_backoff=f"{environment_dt:.4f}s")
        )
        self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)
        self._stub.Ready(services_pb2.Empty(), timeout=self._cfg.connect_timeout_s)

        policy_config = RemotePolicyConfig(
            self._cfg.policy_type,
            self._cfg.pretrained_name_or_path,
            map_robot_keys_to_lerobot_features(self._robot),
            self._cfg.actions_per_chunk,
            self._cfg.policy_device,
        )
        # server loads the model during this call — allow it time
        self._stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(policy_config)), timeout=120.0
        )
        logger.info("Policy server ready at %s (%s)", self._cfg.server_address, self._cfg.policy_type)

    def ping(self) -> bool:
        from lerobot.transport import services_pb2

        if self._stub is None:
            return False
        try:
            self._stub.Ready(services_pb2.Empty(), timeout=self._cfg.connect_timeout_s)
            return True
        except Exception:
            return False

    def send_observation(self, raw_observation: dict[str, Any], timestep: int, must_go: bool) -> bool:
        import pickle  # nosec

        import grpc

        from lerobot.async_inference.helpers import TimedObservation
        from lerobot.transport import services_pb2
        from lerobot.transport.utils import send_bytes_in_chunks

        observation = TimedObservation(
            timestamp=time.time(), observation=raw_observation, timestep=timestep, must_go=must_go
        )
        try:
            iterator = send_bytes_in_chunks(
                pickle.dumps(observation), services_pb2.Observation, log_prefix="[pick_stack] obs", silent=True
            )
            self._stub.SendObservations(iterator)
            return True
        except grpc.RpcError as e:
            logger.error("send_observation failed: %s", e)
            return False

    def poll_actions(self) -> list[ActionStep]:
        import pickle  # nosec

        import grpc

        from lerobot.transport import services_pb2

        try:
            chunk = self._stub.GetActions(services_pb2.Empty())
        except grpc.RpcError as e:
            logger.error("poll_actions failed: %s", e)
            return []
        if len(chunk.data) == 0:
            return []
        timed_actions = pickle.loads(chunk.data)  # nosec - trusted policy server
        return [(ta.get_timestep(), ta.get_action().tolist()) for ta in timed_actions]

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
