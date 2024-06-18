# removing previously acquired results
rm -r ../recon_data/masked_tune_plstv_recons_ood
rm -r ../recon_data/masked_test_plstv_recons_ood

python bart_plstv.py --dist-type 'ood' --idx 0
python bart_plstv.py --dist-type 'ood' --idx 1
python bart_plstv.py --dist-type 'ood' --idx 2
python bart_plstv.py --dist-type 'ood' --idx 3
python bart_plstv.py --dist-type 'ood' --idx 4
mv masked_test_plstv_recons_ood ../recon_data/
mkdir ../recon_data/masked_tune_plstv_recons_ood
mv ../recon_data/masked_test_plstv_recons_ood/recon_0.png ../recon_data/masked_tune_plstv_recons_ood # recon_0 is used as tuning set for sFRC threshold
