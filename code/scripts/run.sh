# Run tsne original and tsne in chain for a dataset
# Then calculate some metrics, generate the scatter plot
# and generate html page to show all result
# See the config of hyper params in icommon.py

DATASET_NAME="COIL20"

python run_multicore-tsne.py -d $DATASET_NAME

python run_chain_perp.py -d $DATASET_NAME

# run compare only for copying file
python run_compare2.py -d $DATASET_NAME

python run_metrics.py -d $DATASET_NAME

# python run_plots.py -d $DATASET_NAME

# python render_view2.py -d $DATASET_NAME
