import gc
import pytest
import sys
import torch
from pytest_benchmark.plugin import BenchmarkFixture, BenchmarkSession, NameWrapper


class CUDABenchmarkFixture(BenchmarkFixture):
    def _make_runner(self, function_to_benchmark, args, kwargs):
        def runner(loops_range, timer=self._timer):
            gc_enabled = gc.isenabled()
            if self._disable_gc:
                gc.disable()
            tracer = sys.gettrace()
            sys.settrace(None)
            try:
                if loops_range:
                    torch.cuda.synchronize()
                    start = timer()

                    for _ in loops_range:
                        function_to_benchmark(*args, **kwargs)

                    torch.cuda.synchronize()
                    end = timer()

                    return end - start
                else:
                    torch.cuda.synchronize()
                    start = timer()

                    result = function_to_benchmark(*args, **kwargs)
                    torch.cuda.synchronize()
                    end = timer()

                    return end - start, result
            finally:
                sys.settrace(tracer)
                if gc_enabled:
                    gc.enable()

        return runner


@pytest.fixture
def cuda_benchmark(request):
    bs: BenchmarkSession = request.config._benchmarksession

    if bs.skip:
        pytest.skip('Benchmarks are skipped (--benchmark-skip was used).')
    else:
        node = request.node
        marker = node.get_closest_marker('benchmark')
        options: dict[str, object] = dict(marker.kwargs) if marker else {}
        if 'timer' in options:
            options['timer'] = NameWrapper(options['timer'])
        fixture = CUDABenchmarkFixture(
                node,
                add_stats=bs.benchmarks.append,
                logger=bs.logger,
                warner=request.node.warn,
                disabled=bs.disabled,
                **dict(bs.options, **options),
        )
        yield fixture
        fixture._cleanup()
