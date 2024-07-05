#!/bin/bash

# python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_A.csv ~/Desktop/rescaled_645nm_per_pixel/A/*.tif
# python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/A/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/A/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_A.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_B.csv ~/Desktop/rescaled_645nm_per_pixel/B/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/B/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/B/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_B.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_C.csv ~/Desktop/rescaled_645nm_per_pixel/C/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/C/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/C/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_C.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_D.csv ~/Desktop/rescaled_645nm_per_pixel/D/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/D/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/D/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_D.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_E.csv ~/Desktop/rescaled_645nm_per_pixel/E/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/E/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/E/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_E.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_F.csv ~/Desktop/rescaled_645nm_per_pixel/F/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/F/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/F/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_F.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_G.csv ~/Desktop/rescaled_645nm_per_pixel/G/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/G/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/G/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_G.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_H.csv ~/Desktop/rescaled_645nm_per_pixel/H/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/H/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/H/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_H.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_I.csv ~/Desktop/rescaled_645nm_per_pixel/I/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/I/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/I/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_I.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_J.csv ~/Desktop/rescaled_645nm_per_pixel/J/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/J/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/J/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_J.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_K.csv ~/Desktop/rescaled_645nm_per_pixel/K/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/K/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/K/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_K.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_L.csv ~/Desktop/rescaled_645nm_per_pixel/L/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/L/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/L/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_L.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_M.csv ~/Desktop/rescaled_645nm_per_pixel/M/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/M/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/M/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_M.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_N.csv ~/Desktop/rescaled_645nm_per_pixel/N/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/N/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/N/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_N.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_O.csv ~/Desktop/rescaled_645nm_per_pixel/O/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/O/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/O/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_O.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_P.csv ~/Desktop/rescaled_645nm_per_pixel/P/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/P/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/P/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_P.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_Q.csv ~/Desktop/rescaled_645nm_per_pixel/Q/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/Q/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/Q/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_Q.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_R.csv ~/Desktop/rescaled_645nm_per_pixel/R/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/R/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/R/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_R.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_S.csv ~/Desktop/rescaled_645nm_per_pixel/S/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/S/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/S/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_S.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_T.csv ~/Desktop/rescaled_645nm_per_pixel/T/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/T/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/T/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_T.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_U.csv ~/Desktop/rescaled_645nm_per_pixel/U/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/U/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/U/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_U.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_V.csv ~/Desktop/rescaled_645nm_per_pixel/V/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/V/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/V/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_V.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_W.csv ~/Desktop/rescaled_645nm_per_pixel/W/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/W/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/W/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_W.csv

python predict.py ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_fasterrcnn_640/ -r ~/RANCZLAB-NAS/iakov/pnns/pnn_v2_scoring_rank_learning/ -t 0.0 -d cuda:0 -o ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_X.csv ~/Desktop/rescaled_645nm_per_pixel/X/*.tif
python draw_predictions.py --root ~/Desktop/rescaled_645nm_per_pixel/X/ --output ~/Desktop/pnns_drawn_predictions_05_07_2024/X/ ~/Desktop/localizations_csv_05_07_2024/localizations_thr_0_data_X.csv