### Cube3

###### Train cost-to-go function
python ctg_approx/avi.py --env cube3 --states_per_update 50000000 --batch_size 10000 --nnet_name cube3 --max_itrs 1000000 --loss_thresh 0.06 --back_max 30 --num_update_procs 30
cp -r saved_models/cube3/current/* saved_models/cube3/target/  # manually update target network
python ctg_approx/avi.py --env cube3 --states_per_update 50000000 --batch_size 10000 --nnet_name cube3 --max_itrs 1200000 --loss_thresh 0.06 --back_max 30 --num_update_procs 30

###### Solve with A* search, use --verbose for more information
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3/current/ --env cube3 --weight 0.6 --batch_size 10000 --results_dir results/cube3/ --language cpp --nnet_batch_size 10000

###### Compare solutions to shortest path
python scripts/compare_solutions.py --soln1 data/cube3/test/data_0.pkl --soln2 results/cube3/results.pkl