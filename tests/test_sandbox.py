import pandas as pd
import pytest

from src.tools.code_executor import CodeExecutor

ATTACKS = {
    "import_os": "import os\nprint(os.listdir('/'))",
    "dunder_import": "__import__('os').system('id')",
    "getattr_import": "getattr(__builtins__, '__import__')('os')",
    "pd_io_os": "print(pd.io.common.os.getcwd())",
    "eval": "eval('1+1')",
    "exec": "exec('x=1')",
    "open": "open('/etc/passwd').read()",
    "subclasses": "print(().__class__.__bases__[0].__subclasses__())",
    "frame": "print((lambda: 1).__globals__)",
    "format_trav": "print('{0.__class__}'.format(1))",
    "to_csv": "df.to_csv('/tmp/x.csv')",
    "read_csv": "pd.read_csv('/etc/passwd')",
    "bare_except": "try:\n    while True: pass\nexcept:\n    pass",
}


@pytest.fixture
def ex():
    return CodeExecutor(timeout_seconds=2)


@pytest.mark.parametrize("name", ATTACKS)
def test_attack_blocked(ex, name):
    r = ex.execute(ATTACKS[name], {"df": pd.DataFrame({"x": [1]}), "pd": pd})
    assert not r.success
    assert r.error_type in ("SecurityError", "SyntaxError"), (name, r.error_type, r.error)


LEGIT = [
    "print(df['x'].mean())",
    "m = getattr(df, 'shape')\nprint(m)",
    "try:\n    df['nope']\nexcept KeyError as e:\n    print('caught')",
    "k = 3\nprint('{} items'.format('a'))\nprint(f'{k}')",
    "print(df.query(f'x > {1}').shape)",
    "import matplotlib.pyplot as plt\nplt.plot(df['x'])\nprint('ok')",
]


@pytest.mark.parametrize("code", LEGIT)
def test_legit_runs(ex, df, code):
    import matplotlib.pyplot as plt

    r = ex.execute(code, {"df": df, "pd": pd, "plt": plt})
    assert r.success, r.error


def test_infinite_loop_times_out(ex, df):
    r = ex.execute("while True:\n    pass", {"df": df})
    assert not r.success and "imeout" in (r.error_type + (r.error or ""))


def test_errors_surface_not_masked(ex, df):
    r = ex.execute("df['missing']", {"df": df})
    assert not r.success and r.error_type == "KeyError"


def test_timeout_also_works_off_main_thread(df):
    import concurrent.futures as cf

    ex = CodeExecutor(timeout_seconds=1)
    with cf.ThreadPoolExecutor(1) as pool:
        r = pool.submit(ex.execute, "while True:\n    pass", {"df": df}).result(timeout=15)
        ok = pool.submit(ex.execute, "print(df['x'].sum())", {"df": df}).result(timeout=15)
    assert not r.success and "imeout" in (r.error_type + (r.error or ""))
    assert ok.success
