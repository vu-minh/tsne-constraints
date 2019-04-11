# Run tsne original and tsne in chain for a dataset
# Then calculate some metrics, generate the scatter plot
# and generate html page to show all result
# See the config of hyper params in icommon.py

DATASET_NAME="COIL20"

# step1: run tsne normal
python run_multicore-tsne.py -d $DATASET_NAME


# step2: run tsne in chain
# it will use some pre-calculated embeddings (corresponding to `base_perps`) in step1
python run_chain_perp.py -d $DATASET_NAME


# step3: run kl compare (KL[Qbase||Q])
# but now use this step only for copying file
python run_compare2.py -d $DATASET_NAME


# step4a: run metric for tSNE normal, so do not specify base_perp
python run_metrics.py -d $DATASET_NAME

# step4b: run metric for tSNE chain with list base_perps defined in the config
python run_metrics.py -d $DATASET_NAME -bp 0


# other steps: create scatter plots and create html page to show them all

# python run_plots.py -d $DATASET_NAME
# python render_view2.py -d $DATASET_NAME
