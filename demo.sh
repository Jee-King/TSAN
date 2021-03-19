#TSAN for train
cd Train/

# TSAN x2  LR: 48 * 48  HR: 96 * 96
python main.py --template TSAN --save TSAN_X2 --scale 2 --reset --save_results --patch_size 96 --ext sep_reset

# TSAN x3  LR: 48 * 48  HR: 144 * 144
python main.py --template TSAN --save TSAN_X3 --scale 3 --reset --save_results --patch_size 144 --ext sep_reset

# TSAN x4  LR: 48 * 48  HR: 192 * 192
python main.py --template TSAN --save TSAN_X4 --scale 4 --reset --save_results --patch_size 192 --ext sep_reset




TSAN for test
cd Test/code/


#TSAN x2
python main.py --data_test MyImage --scale 2 --model TSAN --pre_train ../model/TSAN_x2.pt --test_only --save_results --chop --save "TSAN" --testpath ../LR/LRBI --testset Set5

#TSAN+ x2
python main.py --data_test MyImage --scale 2 --model TSAN --pre_train ../model/TSAN_x2.pt --test_only --save_results --chop --self_ensemble --save "TSAN_plus" --testpath ../LR/LRBI --testset Set5


#TSAN x3
python main.py --data_test MyImage --scale 3 --model TSAN --pre_train ../model/TSAN_x3.pt --test_only --save_results --chop --save "TSAN" --testpath ../LR/LRBI --testset Set5

#TSAN+ x3
python main.py --data_test MyImage --scale 3 --model TSAN --pre_train ../model/TSAN_x3.pt --test_only --save_results --chop --self_ensemble --save "TSAN_plus" --testpath ../LR/LRBI --testset Set5


#TSAN x4
python main.py --data_test MyImage --scale 4 --model TSAN --pre_train ../model/TSAN_x4.pt --test_only --save_results --chop --save "TSAN" --testpath ../LR/LRBI --testset Set5

#TSAN+ x4
python main.py --data_test MyImage --scale 4 --model TSAN --pre_train ../model/TSAN_x4.pt --test_only --save_results --chop --self_ensemble --save "TSAN_plus" --testpath ../LR/LRBI --testset Set5

