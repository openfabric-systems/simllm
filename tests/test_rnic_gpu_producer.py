import pytest

from simllm.compute import (
    GpuTaskKind,
    MemorySpace,
    PipelineKind,
    RnicProducerCoupling,
    RnicProducerRequest,
    RnicProducerShape,
    SmSchedulerModel,
    h100_sxm_80gb_seed_profile,
    rnic_submission_producer_task,
)


def request(
    shape: RnicProducerShape,
    *,
    task_id: str = "producer",
    baseline: int = 10_000,
) -> RnicProducerRequest:
    return RnicProducerRequest(
        task_id=task_id,
        producer_shape=shape,
        wqe_count=1,
        submitted_cycle=2,
        eligible_cycle=3,
        baseline_submission_cycle=baseline,
    )


def test_host_cpu_producer_is_compute_free():
    producer = request(RnicProducerShape.HOST_CPU_DRIVER)

    assert rnic_submission_producer_task(producer) is None
    schedule = RnicProducerCoupling(enabled=True).schedule((producer,))
    assert schedule.estimate is None
    assert schedule.entries[0].producer_task is None
    assert schedule.entries[0].effective_submission_cycle == 10_000


@pytest.mark.parametrize(
    ("shape", "opcodes", "hbm_bytes"),
    (
        (RnicProducerShape.CPU_PROXY, ["STG", "CONTROL"], 64),
        (RnicProducerShape.GPU_INITIATED, ["STG", "STG", "CONTROL"], 68),
    ),
)
def test_non_host_builder_emits_one_network_cta(shape, opcodes, hbm_bytes):
    task = rnic_submission_producer_task(request(shape))

    assert task is not None
    assert task.kind is GpuTaskKind.NETWORK
    assert task.submitted_cycle == 2
    assert task.eligible_cycle == 3
    assert task.launch.grid_blocks == 1
    assert task.launch.threads_per_block == 32
    assert task.launch.static_shared_memory_bytes == 1
    instructions = task.launch.cta_traces[0].warp_traces[0].instructions
    assert [instruction.opcode for instruction in instructions] == opcodes
    assert sum(
        instruction.transacted_bytes
        for instruction in instructions
        if instruction.memory_space is MemorySpace.HBM
    ) == hbm_bytes
    assert all(
        instruction.pipeline is PipelineKind.LOAD_STORE
        for instruction in instructions[:-1]
    )
    assert instructions[-1].pipeline is PipelineKind.CONTROL


def test_disabled_coupling_preserves_deadlines_and_builds_no_tasks():
    requests = (
        request(RnicProducerShape.CPU_PROXY, task_id="proxy", baseline=17),
        request(RnicProducerShape.GPU_INITIATED, task_id="gpu", baseline=19),
    )

    schedule = RnicProducerCoupling().schedule(requests)

    assert schedule.estimate is None
    assert [entry.effective_submission_cycle for entry in schedule.entries] == [
        17,
        19,
    ]
    assert all(entry.producer_task is None for entry in schedule.entries)


def test_enabled_coupling_projects_compute_owned_task_timestamps():
    scheduler = SmSchedulerModel(h100_sxm_80gb_seed_profile())
    producer = request(
        RnicProducerShape.GPU_INITIATED,
        baseline=10_000,
    )

    schedule = RnicProducerCoupling(
        enabled=True,
        scheduler=scheduler,
    ).schedule((producer,))

    assert schedule.estimate is not None
    entry = schedule.entries[0]
    link = entry.producer_task
    assert link is not None
    estimate = next(
        task for task in schedule.estimate.tasks if task.task_id == producer.task_id
    )
    assert (
        link.submitted_cycle,
        link.eligible_cycle,
        link.started_cycle,
        link.completed_cycle,
    ) == (
        estimate.submitted_cycle,
        estimate.eligible_cycle,
        estimate.admitted_cycle,
        estimate.completion_cycle,
    )
    assert link.finished_cycle == link.completed_cycle
    assert entry.effective_submission_cycle == max(
        producer.baseline_submission_cycle,
        link.completed_cycle,
    )


def test_enabled_non_host_coupling_requires_a_scheduler():
    producer = request(RnicProducerShape.CPU_PROXY)

    with pytest.raises(ValueError, match="requires a scheduler"):
        RnicProducerCoupling(enabled=True).schedule((producer,))


def test_producer_request_rejects_nonmonotonic_release():
    with pytest.raises(ValueError, match="eligible_cycle"):
        RnicProducerRequest(
            task_id="backward",
            producer_shape=RnicProducerShape.GPU_INITIATED,
            wqe_count=1,
            submitted_cycle=3,
            eligible_cycle=2,
            baseline_submission_cycle=4,
        )
