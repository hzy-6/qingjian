"""缺词发现器的隐私与撤销语义回归测试。"""

import tempfile
import unittest
from pathlib import Path

from collect_open_lexicons import committed_word_counts


class CommittedWordCountsTest(unittest.TestCase):
    def test_counts_only_unretracted_word_commits_across_sessions(self) -> None:
        rows = (
            '{"event":"session"}\n'
            '{"event":"commit","id":1,"source":"word","text":"甲词"}\n'
            '{"event":"commit","id":2,"source":"sentence","text":"乙句"}\n'
            '{"event":"retract","of":1}\n'
            '{"event":"session"}\n'
            '{"event":"commit","id":1,"source":"word","text":"丙词"}\n'
            '{broken json}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input-log.jsonl"
            path.write_text(rows, encoding="utf-8")
            self.assertEqual(committed_word_counts(path), {"丙词": 1})


if __name__ == "__main__":
    unittest.main()
