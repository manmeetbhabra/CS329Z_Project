"""Reviewer programs. Each is a dspy.Module whose forward(case_dir) returns
dspy.Prediction(flags=[...report flags...], report={...}).

    replay   V0: hand-written findings from tests/fixtures/ (no AI), to test the whole path
"""
