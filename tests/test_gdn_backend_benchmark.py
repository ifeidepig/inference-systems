import argparse
import unittest

from benchmark_gdn_backend import parse_int_list, token_mismatches


class GDNBackendBenchmarkTest(unittest.TestCase):

    def test_parse_int_list(self) -> None:
        self.assertEqual(parse_int_list("1,3,8"), (1, 3, 8))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_int_list("")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_int_list("1,0")

    def test_token_mismatch_reports_first_divergence_per_request(self) -> None:
        reference = [[1, 2, 3], [4, 5, 6]]
        runs = {
            "torch": [{"token_ids": reference}],
            "cuda": [{"token_ids": [[1, 9, 8], [4, 5, 7]]}],
        }
        self.assertEqual(
            token_mismatches(reference, runs),
            [
                {
                    "backend": "cuda",
                    "run_index": 0,
                    "request_index": 0,
                    "token_index": 1,
                    "expected": 2,
                    "actual": 9,
                },
                {
                    "backend": "cuda",
                    "run_index": 0,
                    "request_index": 1,
                    "token_index": 2,
                    "expected": 6,
                    "actual": 7,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
