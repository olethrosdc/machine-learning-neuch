# ---------------------------------------------------------
# Activity A : Discovery
# ---------------------------------------------------------

from simulator import GWASData

# Generate a dataset.
#
# Each row corresponds to an individual.
# Each column corresponds to a genetic marker.
# y is a disease label.

sim = GWASData()

X, y = sim.generate(
n_individuals=500,
n_features=10000,
)

# YOUR TASK
#
# Find one or more features that appear predictive.
#
# You may use:
# - correlations
# - accuracy
# - logistic regression
# - feature importance
# - anything else
#
# Record:
# - which feature(s) you selected
# - why you selected them
# - how predictive they appear


# ---------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------
#
# Stop here.
#
# We will compare results across groups.
#
# ---------------------------------------------------------
