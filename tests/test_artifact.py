import tempfile
import pathlib

from wnba.python import open_artifact


def test_open_artifact_local_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = pathlib.Path(tmpdir) / "hello.txt"
        with open_artifact(str(p), "w") as fw:
            fw.write("edge bets\n")

        with open_artifact(str(p), "r") as fr:
            content = fr.read()

        assert content == "edge bets\n"