# Train format:
bash run.sh train \
    MODEL \
    DATASET \
    GPU_DEVICE \
    SAVE_ID \
    BATCH_SIZE \
    NEGATIVE_SAMPLE_SIZE \
    HIDDEN_DIM \
    GAMMA \
    ALPHA \
    LEARNING_RATE \
    MAX_STEPS \
    TEST_BATCH_SIZE \
    [EXTRA_FLAGS]
#
bash run.sh train RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
# Best Configuration for RotatE
#
bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16 -de
bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
bash run.sh train RotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8 -de
bash run.sh train RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
bash run.sh train RotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
bash run.sh train RotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries 
bash run.sh train RotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
bash run.sh train RotatE YAGO3-10 0 0 1024 400 500 24.0 1.0 0.0002 100000 4 -de
#
# Best Configuration for pRotatE
#
bash run.sh train pRotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16
bash run.sh train pRotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train pRotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
bash run.sh train pRotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
bash run.sh train pRotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train pRotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train pRotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#
# Best Configuration for TransE
# 
bash run.sh train TransE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16
bash run.sh train TransE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train TransE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
bash run.sh train TransE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
bash run.sh train TransE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train TransE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train TransE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#
# Best Configuration for ComplEx
# 
bash run.sh train ComplEx FB15k 0 0 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002
bash run.sh train ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh train ComplEx wn18 0 0 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001
bash run.sh train ComplEx wn18rr 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005
bash run.sh train ComplEx countries_S1 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
bash run.sh train ComplEx countries_S2 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
bash run.sh train ComplEx countries_S3 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
#
# Best Configuration for DistMult
# 
bash run.sh train DistMult FB15k 0 0 1024 256 2000 500.0 1.0 0.001 150000 16 -r 0.000002
bash run.sh train DistMult FB15k-237 0 0 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001
bash run.sh train DistMult wn18 0 0 512 1024 1000 200.0 1.0 0.001 80000 8 -r 0.00001
bash run.sh train DistMult wn18rr 0 0 512 1024 1000 200.0 1.0 0.002 80000 8 -r 0.000005
bash run.sh train DistMult countries_S1 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
bash run.sh train DistMult countries_S2 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
bash run.sh train DistMult countries_S3 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
#
# Valid format:
bash run.sh valid MODEL DATASET GPU_DEVICE SAVE_ID
#
# Test format:
bash run.sh test MODEL DATASET GPU_DEVICE SAVE_ID
#
# Config format:
bash run.sh config_MODEL_DATASET.yaml
# Example:
bash run.sh config_RotatE_wn18rr.yaml
# Config files are stored in configs/
