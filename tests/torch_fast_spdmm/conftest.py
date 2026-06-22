import pytest
from pytest_benchmark.plugin import BenchmarkSession, NameWrapper

from tests.cuda_benchmark import CUDABenchmarkFixture


@pytest.fixture
def benchmark(request):
    """CUDA-aware replacement for pytest-benchmark's 'benchmark' fixture.
    Scoped to this directory so tests elsewhere still get the original fixture.
    Named 'benchmark' so --benchmark-only recognises these tests."""
    bs: BenchmarkSession = request.config._benchmarksession

    if bs.skip:
        pytest.skip("Benchmarks are skipped (--benchmark-skip was used).")

    node = request.node
    marker = node.get_closest_marker("benchmark")
    options = dict(marker.kwargs) if marker else {}
    if "timer" in options:
        options["timer"] = NameWrapper(options["timer"])

    fixture = CUDABenchmarkFixture(
        node,
        add_stats=bs.benchmarks.append,
        logger=bs.logger,
        warner=node.warn,
        disabled=bs.disabled,
        **dict(bs.options, **options),
    )
    yield fixture
    fixture._cleanup()
