"""One entry point for the complete NextBERT training schedule.

Print the plan by default. Pass --execute to train with the fixed two-lane
schedule: two concurrent jobs per wave, with sequential stages in each lane.
"""
from run_two_lane_training import main


if __name__ == "__main__":
    raise SystemExit(main())
