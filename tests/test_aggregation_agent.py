import unittest

from agents.aggregation_agent import _join_branch_summaries


class JoinBranchSummariesTest(unittest.TestCase):
    def test_places_separator_between_branch_summaries(self):
        separator = "\n" + "-" * 80 + "\n"

        result = _join_branch_summaries(["branch one", "branch two", "branch three"])

        self.assertEqual(
            result,
            f"branch one{separator}branch two{separator}branch three",
        )

    def test_preserves_a_single_branch_summary(self):
        self.assertEqual(_join_branch_summaries(["branch one"]), "branch one")


if __name__ == "__main__":
    unittest.main()
