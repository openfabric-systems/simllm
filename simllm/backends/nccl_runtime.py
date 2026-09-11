"""Source-ordered channel FIFOs and GPU work on one retained packet session.

This first path executes buffered float32 Ring primitives. GPU cycles are
declared profile inputs. Source order is not evidence for proprietary GPU
instruction latency, cache behavior, or a calibrated hardware envelope.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from simllm.backends.htsim_nvlink import NvlinkTransfer
from simllm.backends.nccl_resources import NcclGpuProfile, NcclGpuResources
from simllm.traffic.nccl_program import (
    BUFFER_BYTES,
    LL_CLEAN_MASK,
    NcclPrimitive,
    NcclRingProgram,
    balanced_channels,
    ceil_div,
)


@dataclass(frozen=True)
class NcclExecutionConfig:
    """Explicit opt-in to source execution with declared resource inputs."""

    gpu: NcclGpuProfile
    protocol: str = "LL128"
    channels: int = 4
    warps: int = 20
    connection_mode: str = "buffered"

    def __post_init__(self) -> None:
        if not isinstance(self.gpu, NcclGpuProfile):
            raise TypeError("NCCL execution requires a typed GPU profile")
        if type(self.channels) is not int or not 1 <= self.channels <= 64:
            raise ValueError("NCCL channels must be in 1..64")
        self.program(16 * self.channels, (0, 1))
        if self.gpu.residency(self.warps) < 1:
            raise ValueError("NCCL block does not fit the declared SM resources")

    def program(self, payload: int, ranks: tuple[int, ...]) -> NcclRingProgram:
        return NcclRingProgram(payload, ranks, self.protocol,
                               balanced_channels(payload, self.channels), self.warps,
                               self.connection_mode, communicator="ring:" + ":".join(map(str, ranks)))


@dataclass
class NcclReservation:
    sequence: int
    steps: int
    protocol: str
    useful_bytes: int
    encoded_bytes: int
    operation_id: str
    stripes_visible: set[int] = field(default_factory=set)
    data_visible: bool = False
    ready: bool = False


@dataclass
class NcclConnection:
    """Shared head/step lifecycle with separate per-protocol buffer selection."""

    key: tuple[str, int, int, int, int]
    producer_step: int = 0
    consumer_step: int = 0
    returned_head: int = 0
    cached_head: int = 0
    reservations: dict[int, NcclReservation] = field(default_factory=dict)

    def can_reserve(self, steps: int) -> bool:
        return self.cached_head + 8 >= self.producer_step + steps

    def reserve(self, primitive: NcclPrimitive, protocol: str, operation_id: str) -> NcclReservation:
        if not self.can_reserve(primitive.steps):
            raise RuntimeError("NCCL software FIFO has no reusable capacity")
        sequence = self.producer_step
        if any(sequence < other.sequence + other.steps and other.sequence < sequence + primitive.steps
               for other in self.reservations.values()):
            raise AssertionError("NCCL reservation overlaps a live absolute sequence")
        reservation = NcclReservation(sequence, primitive.steps, protocol, primitive.useful_bytes,
                                      primitive.encoded_bytes, operation_id)
        if primitive.useful_bytes or sequence + primitive.steps > self.returned_head:
            self.reservations[sequence] = reservation
        self.producer_step += primitive.steps
        if self.producer_step - self.consumer_step > 8:
            raise AssertionError("NCCL software slot capacity exceeded")
        return reservation

    def consume(self, sequence: int, *, empty_steps: int = 0) -> NcclReservation | None:
        reservation = self.reservations.get(sequence)
        if sequence != self.consumer_step or (not empty_steps and (reservation is None or not reservation.ready)):
            raise AssertionError("NCCL consumed out of order or before data readiness")
        self.consumer_step += empty_steps if empty_steps else reservation.steps
        return reservation

    def return_head(self, head: int) -> None:
        if not self.returned_head <= head <= self.consumer_step:
            raise AssertionError("NCCL returned an invalid consumer head")
        self.returned_head = head
        for sequence in tuple(self.reservations):
            if sequence + self.reservations[sequence].steps <= head:
                del self.reservations[sequence]


@dataclass(frozen=True)
class NcclExecutionResult:
    execution_id: str
    operation_id: str
    program: NcclRingProgram
    released_at_ps: int
    completed_at_ps: int
    extents: tuple
    useful_network_bytes: int
    protocol_data_bytes: int
    counter_bytes: int
    cleanup_bytes: int
    fifo_events: tuple[dict, ...]
    resource_visits: tuple
    residency_events: tuple[dict, ...]
    block_completions: tuple[tuple[str, int], ...]

    @property
    def duration_ps(self) -> int:
        return self.completed_at_ps - self.released_at_ps

    def validate(self) -> None:
        if self.useful_network_bytes != 2 * (len(self.program.ranks) - 1) * self.program.payload_bytes:
            raise AssertionError("source Ring does not conserve useful network bytes")
        if self.protocol_data_bytes != self.program.encoded_network_bytes:
            raise AssertionError("source stripes do not conserve encoded network bytes")
        if sum(extent.transfer.payload_bytes for extent in self.extents) != (
                self.protocol_data_bytes + self.counter_bytes + self.cleanup_bytes):
            raise AssertionError("physical extents lost protocol or counter bytes")
        if max(at for _, at in self.block_completions) != self.completed_at_ps:
            raise AssertionError("collective completion is not its final GPU block completion")
        if len(self.block_completions) != len(self.program.ranks) * len(self.program.channel_bytes):
            raise AssertionError("collective lost a participating block")
        if any(visit.completed_at_ps > self.completed_at_ps for visit in self.resource_visits):
            raise AssertionError("required GPU work follows collective completion")

    def summary(self) -> dict:
        return {"execution_id": self.execution_id, "operation_id": self.operation_id,
                "source_commit": self.program.source_commit,
                "authority": "nccl_channels_on_physical_retained_v1",
                "evidence_class": "declared_model_not_hardware_measurement",
                "released_at_ps": self.released_at_ps, "completed_at_ps": self.completed_at_ps,
                "payload_bytes": self.program.payload_bytes, "protocol": self.program.protocol,
                "width": len(self.program.ranks), "channel_bytes": self.program.channel_bytes,
                "warps": self.program.warps, "connection_mode": self.program.connection_mode,
                "useful_network_bytes": self.useful_network_bytes,
                "protocol_data_bytes": self.protocol_data_bytes,
                "counter_bytes": self.counter_bytes, "cleanup_bytes": self.cleanup_bytes,
                "fifo_events": self.fifo_events,
                "resource_visits": [asdict(row) for row in self.resource_visits],
                "residency_events": self.residency_events,
                "block_completions": self.block_completions}


class NcclChannelRuntime:
    """Persistent connection ownership; only the packet session advances time."""

    def __init__(self, session, gpu: NcclGpuProfile) -> None:
        if session.binding.fabric.switched:
            raise ValueError("NCCL first source slice requires a direct peer mesh")
        self.session = session
        self.gpu = NcclGpuResources(session, gpu)
        self.connections: dict[tuple, NcclConnection] = {}
        self.results: list[NcclExecutionResult] = []
        self._operations: set[tuple[str, str]] = set()
        self._failed = False

    def connection(self, program, channel, src, dst) -> NcclConnection:
        key = (program.communicator, channel, src, dst, 0)
        return self.connections.setdefault(key, NcclConnection(key))

    def validate_program(self, program: NcclRingProgram) -> None:
        if not isinstance(program, NcclRingProgram) or self._failed:
            raise ValueError("NCCL runtime needs a valid program and a live session")
        if self.gpu.profile.residency(program.warps) < 1:
            raise ValueError("NCCL block cannot reside on the declared GPU")
        fabric = self.session.binding.fabric
        for index, rank in enumerate(program.ranks):
            if not fabric.paths_between(rank, program.ranks[(index + 1) % len(program.ranks)]):
                raise ValueError("NCCL ring edge has no physical route")

    def run(self, execution_id: str, operation_id: str, program: NcclRingProgram,
            *, receiver_delay_ps: int = 0) -> NcclExecutionResult:
        self.validate_program(program)
        if ((execution_id, operation_id) in self._operations
                or any(not isinstance(value, str) or not value for value in (execution_id, operation_id))
                or type(receiver_delay_ps) is not int or receiver_delay_ps < 0):
            raise ValueError("NCCL operation must be unique with a nonnegative receiver delay")
        self._operations.add((execution_id, operation_id))
        operation = _Operation(self, execution_id, operation_id, program, receiver_delay_ps)
        try:
            operation.start()
            self.session.advance_until(lambda: operation.complete)
            result = operation.result()
            result.validate()
        except Exception:
            self._failed = True
            raise
        self.results.append(result)
        return result


class _Operation:
    def __init__(self, runtime, execution_id, operation_id, program, receiver_delay):
        self.runtime, self.session, self.gpu = runtime, runtime.session, runtime.gpu
        self.execution_id, self.operation_id, self.program = execution_id, operation_id, program
        self.release = self.session.now_ps
        self.receiver_delay = receiver_delay
        self.extents, self.fifo_events, self.completions = [], [], []
        self.bytes = {"data": 0, "counter": 0, "cleanup": 0}
        self.visit_start = len(self.gpu.visits)
        self.residency_start = len(self.gpu.residency_events)
        self.serial = 0
        self.blocks = []

    @property
    def complete(self):
        return len(self.completions) == len(self.program.ranks) * len(self.program.channel_bytes)

    def log(self, kind, connection, **fields):
        self.fifo_events.append(dict(at_ps=self.session.now_ps, kind=kind, connection=connection.key,
                                     producer_step=connection.producer_step,
                                     consumer_step=connection.consumer_step,
                                     returned_head=connection.returned_head,
                                     cached_head=connection.cached_head, **fields))

    def transfer(self, source, destination, size, kind, callback):
        self.serial += 1
        name = f"{self.execution_id}:{self.operation_id}:nccl-{self.serial}"
        transfer = NvlinkTransfer(extent_id=name, source=source, destination=destination,
                                  payload_bytes=size, released_at_ps=self.session.now_ps,
                                  topology_endpoint_count=max(self.program.ranks) + 1)
        self.extents.extend(self.session.admit(self.execution_id, self.operation_id, (transfer,)))
        self.bytes[kind] += size
        self.session.on_visible(name, callback)

    def start(self):
        self.session.schedule_callback(
            self.release + self.gpu.profile.cycles_ps(self.gpu.profile.kernel_entry_cycles), self.admit_blocks)

    def admit_blocks(self):
        # Matching block admission order on all participants avoids constructing
        # cross-rank residency deadlocks in this explicitly deterministic policy.
        for channel in range(len(self.program.channel_bytes)):
            for rank_index in range(len(self.program.ranks)):
                block = _Block(self, channel, rank_index)
                self.blocks.append(block)
                self.gpu.admit(block.id, block.rank, self.program.warps, block.start)

    def result(self):
        return NcclExecutionResult(
            self.execution_id, self.operation_id, self.program, self.release,
            max(at for _, at in self.completions), tuple(self.extents),
            self.program.useful_network_bytes, self.bytes["data"], self.bytes["counter"],
            self.bytes["cleanup"], tuple(self.fifo_events), self.gpu.visits[self.visit_start:],
            self.gpu.residency_events[self.residency_start:], tuple(self.completions),
        )


class _Block:
    def __init__(self, operation, channel, rank_index):
        self.op, self.program = operation, operation.program
        self.session, self.gpu, self.profile = operation.session, operation.gpu, operation.gpu.profile
        self.channel, self.rank_index = channel, rank_index
        self.rank = self.program.ranks[rank_index]
        self.next_rank = self.program.ranks[(rank_index + 1) % len(self.program.ranks)]
        self.prev_rank = self.program.ranks[(rank_index - 1) % len(self.program.ranks)]
        self.id = f"{operation.execution_id}:{operation.operation_id}:c{channel}:r{self.rank}"
        self.send_conn = operation.runtime.connection(self.program, channel, self.rank, self.next_rank)
        self.recv_conn = operation.runtime.connection(self.program, channel, self.prev_rank, self.rank)
        self.primitives = self.program.primitives(channel, rank_index)
        self.position = 0
        self.recv_step = self.recv_conn.consumer_step
        self.send_reservation = None
        self.inflight_data = 0
        self.pending_stores = 0
        self.finish_waiting = False
        self.data_fence_callback = None

    def service(self, kind, units, callback, *, memory=False, shared=False):
        self.gpu.service(self.id, kind, units, callback, memory=memory, shared=shared)

    def transfer(self, destination, size, kind, callback):
        self.pending_stores += 1

        def visible():
            self.pending_stores -= 1
            callback()
            if not self.pending_stores and self.finish_waiting:
                self.finish_waiting = False
                self.session.schedule_callback(self.session.now_ps, self.finish)

        self.op.transfer(self.rank, destination, size, kind, visible)

    def start(self):
        self.service("block_setup", self.profile.block_setup_cycles, self.initialize)

    def initialize(self):
        # loadSendSync/loadSendConn refresh the constructor's cached head.
        self.send_conn.cached_head = self.send_conn.returned_head
        self.op.log("constructor_head_load", self.send_conn)
        if self.program.protocol == "SIMPLE":
            # Source constructor rounds both endpoints and publishes recv head.
            self.send_conn.producer_step = ceil_div(self.send_conn.producer_step, 4) * 4
            self.recv_step = ceil_div(self.recv_conn.consumer_step, 4) * 4
            self.recv_conn.consumer_step = self.recv_step
            head = self.recv_step
            self.op.log("constructor_alignment", self.recv_conn, sequence=head)
            self.transfer(self.prev_rank, 8, "counter", lambda: self.recv_conn.return_head(head))
        self.next_primitive()

    def poll(self, kind: str, predicate: Callable[[], bool], callback: Callable[[], None]):
        began = self.session.now_ps

        def checked():
            if predicate():
                callback()
            else:
                self.session.schedule_callback(max(self.session.now_ps, began + self.profile.cycles_ps(
                    self.profile.poll_interval_cycles)), lambda: self.poll(kind, predicate, callback))

        self.service(kind, self.profile.poll_issue_cycles, checked)

    def next_primitive(self):
        if self.position == len(self.primitives):
            self.service("destructor_barrier", self.profile.barrier_cycles_per_warp * self.program.warps,
                         self.finish)
            return
        self.primitive = self.primitives[self.position]
        self.send_reservation = None
        self.inflight_data = 0
        if self.primitive.send:
            def capacity():
                self.send_conn.cached_head = self.send_conn.returned_head
                return self.send_conn.can_reserve(self.primitive.steps)

            def reserve():
                self.send_reservation = self.send_conn.reserve(self.primitive, self.program.protocol,
                                                              self.op.operation_id)
                self.op.log("reserve", self.send_conn, sequence=self.send_reservation.sequence,
                            slot=self.send_reservation.sequence % 8, steps=self.primitive.steps,
                            useful_bytes=self.primitive.useful_bytes,
                            encoded_bytes=self.primitive.encoded_bytes, protocol=self.program.protocol)
                self.begin_primitive()

            if self.send_conn.can_reserve(self.primitive.steps):
                reserve()
            else:
                self.op.log("capacity_wait", self.send_conn, steps=self.primitive.steps)
                self.poll("head_poll", capacity, reserve)
        else:
            self.begin_primitive()

    def begin_primitive(self):
        # LL waitSend carries the entry barrier; LL128 always has it. Simple
        # workers synchronize after waitPeer, before reduceCopy.
        cycles = (self.profile.barrier_cycles_per_warp * self.program.warps
                  if self.program.protocol != "LL" or self.primitive.send else 0)
        self.service("primitive_entry_barrier", cycles, self.begin_stripes)

    def begin_stripes(self):
        if self.primitive.receive and self.rank_index == 0 and self.op.receiver_delay:
            self.session.schedule_callback(self.session.now_ps + self.op.receiver_delay, self.wait_receive)
        else:
            self.wait_receive()

    def wait_receive(self):
        p = self.primitive
        if self.program.protocol == "SIMPLE" and p.receive:
            def ready():
                reservation = self.recv_conn.reservations.get(self.recv_step)
                return reservation is not None and reservation.ready
            self.poll("tail_poll", ready, self.launch_warps)
        else:
            self.launch_warps()

    def launch_warps(self):
        p = self.primitive
        self.warp_queues = {warp: [s for s in p.stripes if s.warp == warp]
                            for warp in {s.warp for s in p.stripes}}
        self.remaining_warps = len(self.warp_queues)
        if not self.remaining_warps:
            # Empty LL/LL128 receives have no data-ready loop in the source.
            # Their head may advance before the matching empty send reserves.
            self.end_primitive()
            return
        for warp in self.warp_queues:
            self.stripe(warp, 0)

    def stripe(self, warp, index):
        if index == len(self.warp_queues[warp]):
            self.remaining_warps -= 1
            if self.remaining_warps == 0:
                self.end_primitive()
            return
        stripe = self.warp_queues[warp][index]
        p = self.primitive
        load_complete = not p.source_load
        receive_complete = False
        compute_scheduled = False

        def ready():
            reservation = self.recv_conn.reservations.get(self.recv_step)
            return reservation is not None and stripe.index in reservation.stripes_visible

        def compute():
            nonlocal receive_complete
            receive_complete = True
            join()

        def loaded():
            nonlocal load_complete
            load_complete = True
            join()

        def join():
            nonlocal compute_scheduled
            if not load_complete or not receive_complete or compute_scheduled:
                return
            compute_scheduled = True
            groups = 4 if self.program.protocol == "LL128" else 1
            work = groups * (1 + p.reduce) * self.profile.warp_issue_cycles
            self.service("reduce_pack" if p.reduce else "pack_copy", work, send)

        def receive():
            if p.receive and self.program.protocol != "SIMPLE":
                self.poll("flag_poll", ready, lambda: self.service(
                    "peer_buffer_load", stripe.encoded_bytes, compute, memory=True))
            elif p.receive:
                self.service("peer_buffer_load", stripe.encoded_bytes, compute, memory=True)
            else:
                compute()

        def sent_visible(reservation, stripe_index):
            reservation.stripes_visible.add(stripe_index)
            if len(reservation.stripes_visible) == len(p.stripes):
                reservation.data_visible = True
                if self.program.protocol != "SIMPLE":
                    reservation.ready = True
                elif self.data_fence_callback is not None:
                    callback = self.data_fence_callback
                    self.data_fence_callback = None
                    self.session.schedule_callback(self.session.now_ps, callback)
            self.inflight_data -= 1

        def send():
            if p.send:
                reservation = self.send_reservation
                self.inflight_data += 1
                self.transfer(self.next_rank, stripe.encoded_bytes, "data",
                              lambda: sent_visible(reservation, stripe.index))
            if not p.output_store:
                done()
            elif self.program.protocol == "LL128":
                output_ll128()
            else:
                self.service("output_store", stripe.useful_bytes, done, memory=True)

        def output_ll128():
            # Aligned global stores and shared staging have distinct service.
            # The warp barrier orders staged tail reads, not a global-memory fence.
            unfinished = 2

            def branch_done():
                nonlocal unfinished
                unfinished -= 1
                if unfinished == 0:
                    done()

            tail = stripe.shared_tail_load_bytes
            aligned = stripe.useful_bytes - tail
            if aligned:
                self.service("output_store", aligned, branch_done, memory=True)
            else:
                branch_done()

            def tail_store():
                self.service("output_store", tail, branch_done, memory=True)

            def tail_read():
                if tail:
                    self.service("shared_tail_load", tail, tail_store, shared=True)
                else:
                    branch_done()

            def warp_barrier():
                self.service("output_warp_barrier", self.profile.barrier_cycles_per_warp, tail_read)

            if stripe.shared_staging_bytes:
                self.service("shared_staging_store", stripe.shared_staging_bytes, warp_barrier, shared=True)
            else:
                warp_barrier()

        def done():
            self.stripe(warp, index + 1)

        if p.source_load:
            # Source loadBegin issues before receive polling; loadFinish joins
            # their completions. Unrelated ready work may execute meanwhile.
            self.service("input_load", stripe.local_load_bytes, loaded, memory=True)
        receive()

    def end_primitive(self):
        p = self.primitive
        if p.send and not p.stripes:
            self.send_reservation.data_visible = True
            if self.program.protocol != "SIMPLE":
                self.send_reservation.ready = True
        self.service("primitive_exit_barrier", self.profile.barrier_cycles_per_warp * self.program.warps,
                     self.publish)

    def publish(self):
        p = self.primitive

        def receive_post():
            if p.receive:
                reservation = self.recv_conn.reservations.get(self.recv_step)
                empty_inline = not p.stripes and self.program.protocol != "SIMPLE"
                if not empty_inline and (reservation is None or not reservation.ready):
                    # All required stripes have been consumed; visibility
                    # callbacks at this timestamp must still settle first.
                    self.poll("completion_flag_poll", lambda: reservation.ready, receive_post)
                    return
                sequence = self.recv_step
                self.recv_conn.consume(sequence, empty_steps=p.steps if empty_inline else 0)
                self.recv_step += p.steps
                self.op.log("consume", self.recv_conn, sequence=sequence,
                            useful_bytes=p.useful_bytes, protocol=self.program.protocol)
                head = self.recv_step

                def returned():
                    self.recv_conn.return_head(head)
                    self.op.log("head_visible", self.recv_conn, sequence=head)

                def issued():
                    self.transfer(self.prev_rank, 8, "counter", returned)
                    cleanup()

                self.service("head_publication", self.profile.publication_cycles, issued)
            else:
                cleanup()

        def cleanup():
            if p.send and self.program.protocol == "LL" and (
                    self.send_reservation.sequence & LL_CLEAN_MASK) == LL_CLEAN_MASK:
                extra = BUFFER_BYTES["LL"] // 8 - p.encoded_bytes
                if extra:
                    self.transfer(self.next_rank, extra, "cleanup", lambda: None)
                self.service("flag_cleanup", ceil_div(extra, 512) * self.profile.warp_issue_cycles,
                             self.advance)
            else:
                self.advance()

        if p.send and self.program.protocol == "SIMPLE":
            reservation = self.send_reservation

            def post():
                def visible():
                    reservation.ready = True
                    self.op.log("tail_visible", self.send_conn, sequence=reservation.sequence + p.steps)
                self.transfer(self.next_rank, 8, "counter", visible)
                receive_post()

            def fence():
                self.service("fence_and_tail_publication" if p.stripes else "tail_publication",
                             self.profile.publication_cycles, post)

            # Separate progress is published after all data is visible. There
            # is no unconditional extra RTT and no separately launched poll kernel.
            if p.stripes:
                if reservation.data_visible:
                    fence()
                else:
                    # Fence readiness is a memory-order dependency, not a
                    # software polling loop invented inside the primitive.
                    self.data_fence_callback = fence
            else:
                fence()
        else:
            receive_post()

    def advance(self):
        self.position += 1
        self.session.schedule_callback(self.session.now_ps, self.next_primitive)

    def finish(self):
        if self.pending_stores:
            self.finish_waiting = True
            return
        self.op.completions.append((self.id, self.session.now_ps))
        self.gpu.release(self.id)
