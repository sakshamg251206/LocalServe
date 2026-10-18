from pathlib import Path

from localserve.server import llama_server_cmd, parse_footprint

SAMPLE = """llama-server [31727]: 64-bit    Footprint: 2635 MB (16384 bytes per page)
2635 MB     622 MB          0 B       3772    TOTAL
    phys_footprint: 2635 MB
    phys_footprint_peak: 2.5 GB
"""


def test_parse_footprint_units() -> None:
    assert parse_footprint(SAMPLE) == {
        "phys_footprint_bytes": 2635 * 1024**2,
        "phys_footprint_peak_bytes": int(2.5 * 1024**3),
    }
    assert parse_footprint("garbage") == {}


def test_llama_server_cmd_splits_context_per_slot() -> None:
    cmd = llama_server_cmd(Path("m.gguf"), "a", parallel=4, ctx_per_slot=2048, port=9)
    assert cmd[cmd.index("-c") + 1] == "8192"
    assert cmd[cmd.index("-np") + 1] == "4"
    assert cmd[cmd.index("-ngl") + 1] == "99"
