#!/bin/sh
DATASET=modcloth
DIM=10
LR=0.001
#LBDA=0.1
#C=1.0
METHOD=MF

PROTECT_ITEM=0
PROTECT_USER=0
PROTECT_USER_ITEM=0

python ./src/main.py --dataset $DATASET --method $METHOD --hidden_dim $DIM --learning_rate $LR \
    --protect_item_group $PROTECT_ITEM --protect_user_group $PROTECT_USER --protect_user_item_group $PROTECT_USER_ITEM \
    >${METHOD}_${PROTECT_ITEM}_${PROTECT_USER}_${PROTECT_USER_ITEM}.log 2>&1 &

# PROTECT_ITEM=1
# PROTECT_USER=0
# PROTECT_USER_ITEM=0

# python ./src/main.py --dataset $DATASET --method $METHOD --hidden_dim $DIM --learning_rate $LR \
#     --protect_item_group $PROTECT_ITEM --protect_user_group $PROTECT_USER --protect_user_item_group $PROTECT_USER_ITEM \
#     >${METHOD}_${PROTECT_ITEM}_${PROTECT_USER}_${PROTECT_USER_ITEM}.log 2>&1 &

# PROTECT_ITEM=0
# PROTECT_USER=1
# PROTECT_USER_ITEM=0

# python ./src/main.py --dataset $DATASET --method $METHOD --hidden_dim $DIM --learning_rate $LR \
#     --protect_item_group $PROTECT_ITEM --protect_user_group $PROTECT_USER --protect_user_item_group $PROTECT_USER_ITEM \
#     >${METHOD}_${PROTECT_ITEM}_${PROTECT_USER}_${PROTECT_USER_ITEM}.log 2>&1 &

# PROTECT_ITEM=0
# PROTECT_USER=0
# PROTECT_USER_ITEM=1

# python ./src/main.py --dataset $DATASET --method $METHOD --hidden_dim $DIM --learning_rate $LR \
#     --protect_item_group $PROTECT_ITEM --protect_user_group $PROTECT_USER --protect_user_item_group $PROTECT_USER_ITEM \
#     >${METHOD}_${PROTECT_ITEM}_${PROTECT_USER}_${PROTECT_USER_ITEM}.log 2>&1 &

# PROTECT_ITEM=1
# PROTECT_USER=1
# PROTECT_USER_ITEM=0

# python ./src/main.py --dataset $DATASET --method $METHOD --hidden_dim $DIM --learning_rate $LR \
#     --protect_item_group $PROTECT_ITEM --protect_user_group $PROTECT_USER --protect_user_item_group $PROTECT_USER_ITEM \
#     >${METHOD}_${PROTECT_ITEM}_${PROTECT_USER}_${PROTECT_USER_ITEM}.log 2>&1 &

# PROTECT_ITEM=1
# PROTECT_USER=1
# PROTECT_USER_ITEM=1

# python ./src/main.py --dataset $DATASET --method $METHOD --hidden_dim $DIM --learning_rate $LR \
#     --protect_item_group $PROTECT_ITEM --protect_user_group $PROTECT_USER --protect_user_item_group $PROTECT_USER_ITEM \
#     >${METHOD}_${PROTECT_ITEM}_${PROTECT_USER}_${PROTECT_USER_ITEM}.log 2>&1 &