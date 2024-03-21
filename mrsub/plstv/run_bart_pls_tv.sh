python bart_plstv.py --dist-type 'ood' --idx 0
python bart_plstv.py --dist-type 'ood' --idx 1
python bart_plstv.py --dist-type 'ood' --idx 2
python bart_plstv.py --dist-type 'ood' --idx 3
python bart_plstv.py --dist-type 'ood' --idx 4
mv masked__plstv_recons_ood ../recon_data/
mkdir ../recon_data/masked_sel_plstv_recons_ood
mv ../recon_data/masked__plstv_recons_ood/recon_0.png ../recon_data/masked_sel_plstv_recons_ood # recon_0 is used as tuning set for sFRC threshold
