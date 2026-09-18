from __future__ import annotations

import math
import threading

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from numba import njit
from pydantic import BaseModel


# ================================================================
# CONFIGURATION
# ================================================================

# This is deliberately much larger than our original prototype.
N_NEURONS = 8_000

# Sparse connectivity.
# Each neuron receives approximately this many outgoing edges.
CONNECTIONS_PER_NEURON = 25

# Neural simulation rate.
NEURAL_HZ = 50.0

DT = 1.0 / NEURAL_HZ

MEMBRANE_DECAY = 0.92

THRESHOLD = 1.0

RESET = 0.0

SYNAPSE_SCALE = 0.035


# ================================================================
# CREATE A LARGE SPARSE NETWORK
# ================================================================

rng = np.random.default_rng(12345)

TOTAL_CONNECTIONS = (
    N_NEURONS *
    CONNECTIONS_PER_NEURON
)

print(
    "Creating neural network:",
    N_NEURONS,
    "neurons"
)

print(
    "Connections:",
    TOTAL_CONNECTIONS
)


# Every neuron gets CONNECTIONS_PER_NEURON
# outgoing edges.

source_indices = np.repeat(
    np.arange(
        N_NEURONS,
        dtype=np.int32
    ),
    CONNECTIONS_PER_NEURON
)

target_indices = rng.integers(
    0,
    N_NEURONS,
    size=TOTAL_CONNECTIONS,
    dtype=np.int32
)

# Signed synapses.
#
# Positive = excitatory
# Negative = inhibitory
#
# This is NOT yet the actual FlyWire transmitter map.
signs = np.where(
    rng.random(TOTAL_CONNECTIONS) < 0.8,
    1.0,
    -1.0,
).astype(np.float32)

weights = (
    signs *
    rng.uniform(
        0.25,
        1.0,
        TOTAL_CONNECTIONS
    ).astype(np.float32)
)


# ================================================================
# CSR-LIKE STRUCTURE
# ================================================================

indptr = np.arange(
    0,
    TOTAL_CONNECTIONS + 1,
    CONNECTIONS_PER_NEURON,
    dtype=np.int32
)


# ================================================================
# GLOBAL BRAIN STATE
# ================================================================

class Brain:
    def __init__(self):
        self.voltage = np.zeros(
            N_NEURONS,
            dtype=np.float32
        )

        self.synaptic = np.zeros(
            N_NEURONS,
            dtype=np.float32
        )

        self.spikes = np.zeros(
            N_NEURONS,
            dtype=np.uint8
        )

        self.lock = threading.Lock()


brains: dict[str, Brain] = {}

brains_lock = threading.Lock()


def get_brain(fly_id: str) -> Brain:
    with brains_lock:

        if fly_id not in brains:
            brains[fly_id] = Brain()

        return brains[fly_id]


# ================================================================
# NEURAL STEP
# ================================================================

@njit(cache=True)
def neural_step(
    voltage,
    synaptic,
    spikes,
    target_indices,
    source_indices,
    weights,
    sensory,
):
    # ------------------------------------------------------------
    # membrane integration
    # ------------------------------------------------------------

    for i in range(N_NEURONS):

        voltage[i] *= MEMBRANE_DECAY

        voltage[i] += synaptic[i]

        synaptic[i] = 0.0

        spikes[i] = 0

    # ------------------------------------------------------------
    # SENSORY INPUT
    #
    # The first part of the neural population is treated as
    # sensory neurons for this prototype.
    # ------------------------------------------------------------

    sensory_count = sensory.shape[0]

    for i in range(sensory_count):

        target =
            int(
                i *
                N_NEURONS /
                sensory_count /
                4
            )

        if target >= N_NEURONS:
            target = N_NEURONS - 1

        voltage[target] += (
            sensory[i] *
            0.65
        )

    # ------------------------------------------------------------
    # SPIKING
    # ------------------------------------------------------------

    for i in range(N_NEURONS):

        if voltage[i] >= THRESHOLD:

            spikes[i] = 1

            voltage[i] = RESET

    # ------------------------------------------------------------
    # PROPAGATION
    # ------------------------------------------------------------

    for edge in range(
        target_indices.shape[0]
    ):

        source = source_indices[edge]

        if spikes[source]:

            destination =
                target_indices[edge]

            synaptic[destination] += (
                weights[edge] *
                SYNAPSE_SCALE
            )


# ================================================================
# MOTOR POPULATIONS
# ================================================================

# The actual FlyWire motor populations will replace these later.
#
# For now these are distributed populations so that behavior
# emerges from the recurrent network rather than one hard-coded
# direction.

MOTOR_FORWARD = np.arange(
    6500,
    7000,
    dtype=np.int32
)

MOTOR_TURN_LEFT = np.arange(
    7000,
    7300,
    dtype=np.int32
)

MOTOR_TURN_RIGHT = np.arange(
    7300,
    7600,
    dtype=np.int32
)

MOTOR_LIFT = np.arange(
    7600,
    7800,
    dtype=np.int32
)

MOTOR_BRAKE = np.arange(
    7800,
    8000,
    dtype=np.int32
)


def population_activity(
    brain: Brain,
    population: np.ndarray
) -> float:

    if len(population) == 0:
        return 0.0

    voltage =
        brain.voltage[population]

    spikes =
        brain.spikes[population]

    return float(
        np.mean(voltage) +
        np.mean(spikes) *
        0.5
    )


# ================================================================
# ROBLOX API
# ================================================================

app = FastAPI()


class StepRequest(BaseModel):

    fly_id: str

    sensors: list[list[float]]


@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Fly Neural Server",
        "neurons": N_NEURONS,
        "connections": TOTAL_CONNECTIONS
    }


@app.post("/step")
def step(
    request: StepRequest,
    x_api_key: str | None = Header(
        default=None
    )
):

    brain = get_brain(
        request.fly_id
    )

    outputs = []

    with brain.lock:

        for sensory_sample in request.sensors:

            sensory =
                np.asarray(
                    sensory_sample,
                    dtype=np.float32
                )

            # Clamp incoming values.
            sensory =
                np.clip(
                    sensory,
                    -1.0,
                    1.0
                )

            # ----------------------------------------------------
            # Run multiple internal neural updates.
            # ----------------------------------------------------

            for _ in range(3):

                neural_step(
                    brain.voltage,
                    brain.synaptic,
                    brain.spikes,
                    target_indices,
                    source_indices,
                    weights,
                    sensory
                )

            # ----------------------------------------------------
            # MOTOR READOUT
            # ----------------------------------------------------

            forward =
                population_activity(
                    brain,
                    MOTOR_FORWARD
                )

            left =
                population_activity(
                    brain,
                    MOTOR_TURN_LEFT
                )

            right =
                population_activity(
                    brain,
                    MOTOR_TURN_RIGHT
                )

            lift =
                population_activity(
                    brain,
                    MOTOR_LIFT
                )

            brake =
                population_activity(
                    brain,
                    MOTOR_BRAKE
                )

            # ----------------------------------------------------
            # Convert population activity into motor signals.
            # ----------------------------------------------------

            turn =
                right - left

            forward =
                math.tanh(
                    forward * 3.0
                )

            turn =
                math.tanh(
                    turn * 5.0
                )

            lift =
                math.tanh(
                    lift * 4.0
                )

            brake =
                1.0 / (
                    1.0 +
                    math.exp(
                        -brake * 4.0
                    )
                )

            outputs.append({
                "forward": float(
                    forward
                ),

                "turn": float(
                    turn
                ),

                "lift": float(
                    lift
                ),

                "brake": float(
                    brake
                )
            })

    return {
        "outputs": outputs,
        "neurons": N_NEURONS,
        "connections":
            TOTAL_CONNECTIONS
    }
