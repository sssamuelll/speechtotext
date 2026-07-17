import subprocess
import sys

from speechtotext.statistics import (
    one_sided_error_upper,
    one_sided_success_lower,
)


# Codigo que corre en un subprocess limpio con TODO import de scipy bloqueado,
# probando que speechtotext.statistics no lo necesita y que las colas cerradas,
# la monotonicidad y el determinismo bit a bit se cumplen sin la referencia.
_SUBPROCESS = r"""
import importlib.abc
import sys


class _BlockSciPy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy prohibido en runtime")
        return None


sys.meta_path.insert(0, _BlockSciPy())

import speechtotext.statistics as stats

assert "scipy" not in sys.modules, "runtime importo scipy"

# Casos cerrados exactos.
assert stats.one_sided_success_lower(0, 5) == 0.0
assert stats.one_sided_error_upper(5, 5) == 1.0
tail = stats.one_sided_error_upper(0, 4)
assert abs(tail - (1.0 - 0.05 ** (1.0 / 4))) < 1e-15
full = stats.one_sided_success_lower(4, 4)
assert abs(full - 0.05 ** (1.0 / 4)) < 1e-15

# Monotonicidad: mas exitos sobre el mismo n sube el limite inferior.
lowers = [stats.one_sided_success_lower(k, 10) for k in range(11)]
assert lowers == sorted(lowers)
assert lowers[0] == 0.0 and 0.0 < lowers[-1] < 1.0

# Determinismo bit a bit: la biseccion no depende de estado global.
a = stats.one_sided_success_lower(7, 10)
b = stats.one_sided_success_lower(7, 10)
assert a.hex() == b.hex()

print("STATISTICS_RUNTIME_OK")
"""


def test_statistics_runtime_no_requiere_scipy():
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "STATISTICS_RUNTIME_OK" in completed.stdout


def test_colas_cerradas_en_proceso():
    assert one_sided_success_lower(0, 7) == 0.0
    assert one_sided_error_upper(7, 7) == 1.0


def test_monotonicidad_y_determinismo():
    lowers = [one_sided_success_lower(k, 20) for k in range(21)]
    assert lowers == sorted(lowers)
    assert one_sided_success_lower(13, 20).hex() == (
        one_sided_success_lower(13, 20).hex()
    )
