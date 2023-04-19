import time
import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load, run_result_show
from src.data import context_data_load, context_data_split, context_data_loader, context_data_stratified_kfold_split
from src.data import modified_context_data_load
from src.data import dl_data_load, dl_data_split, dl_data_loader, dl_data_stratified_kfold_split
from src.data import image_data_load, image_data_split, image_data_loader, image_data_stratified_kfold_split
from src.data import text_data_load, text_data_split, text_data_loader, text_data_stratified_kfold_split
from src.train import train, test

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset



from sklearn.model_selection import StratifiedKFold


def main(args):
    Setting.seed_everything(args.seed)


    ######################## DATA LOAD
    print(f'--------------- {args.model} Load Data ---------------')
    if args.model in ('FM', 'FFM'):
        data = context_data_load(args)
        if args.context_data!='baseline':
            data = modified_context_data_load(args)
    elif args.model in ('NCF', 'WDN', 'DCN'):
        data = dl_data_load(args)
    elif args.model == 'CNN_FM':
        data = image_data_load(args)
    elif args.model == 'DeepCoNN':
        import nltk
        nltk.download('punkt')
        data = text_data_load(args)
    else:
        pass


    ######################## Train/Valid Split
    print(f'--------------- {args.model} Train/Valid Split ---------------')

    kfold = args.kfold
    n_splits = args.kfold_n_splits
    
    if kfold != 0:
        # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        if args.model in ('FM', 'FFM'):
            indices, base_data = context_data_stratified_kfold_split(args, data)
        elif args.model in ('NCF', 'WDN', 'DCN'):
            indices, base_data = dl_data_stratified_kfold_split(args, data)
        elif args.model=='CNN_FM':
            indices, base_data = image_data_stratified_kfold_split(args, data)
        elif args.model=='DeepCoNN':
            indices, base_data = text_data_stratified_kfold_split(args, data)
        else:
            pass


    # if args.model in ('FM', 'FFM'):
    #     data = context_data_split(args, data)
    #     data = context_data_loader(args, data)
    # elif args.model in ('NCF', 'WDN', 'DCN'):
    #     data = dl_data_split(args, data)
    #     data = dl_data_loader(args, data)

    # elif args.model=='CNN_FM':
    #     data = image_data_split(args, data)
    #     data = image_data_loader(args, data)

    # elif args.model=='DeepCoNN':
    #     data = text_data_split(args, data)
    #     data = text_data_loader(args, data)
    # else:
    #     pass

    ####################### Setting for Log
    setting = Setting()

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)
    logger = Logger(args, log_path)
    logger.save_args()
    
    predict_avg_result = None
    for fold, (train_indices, val_indices) in enumerate(indices):
        print(f'--------------- Fold [{fold+1}/{n_splits}] Running...')

        print('create dataloader')

        data['X_train'] = base_data.iloc[train_indices].drop(['rating'], axis=1)
        data['X_valid'] = base_data.iloc[val_indices].drop(['rating'], axis=1)
        data['y_train'] = base_data.iloc[train_indices]['rating']
        data['y_valid'] = base_data.iloc[val_indices]['rating']

        train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
        valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
        test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader


        ######################## Model
        print(f'--------------- INIT {args.model} ---------------')
        model = models_load(args,data)


        ######################## TRAIN
        print(f'--------------- {args.model} TRAINING ---------------')
        model,min_loss = train(args, model, data, logger, setting)


        ######################## INFERENCE
        print(f'--------------- {args.model} PREDICT ---------------')
        predicts = test(args, model, data, setting)
        
        if predict_avg_result == None:
            predict_avg_result = predicts   # First fold of prediction
        else:
            predict_avg_result += predicts  # Add predicts result
        print(f'--------------- Fold [{fold+1}/{n_splits}] Finish!')
        print(f'--------------------------------------------------')
    
    predict_avg_result /= n_splits  # average
    
    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    if args.model in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN'):
        submission['rating'] = predicts
    else:
        pass

    filename = setting.get_submit_filename(args,min_loss)
    submission.to_csv(filename, index=False)
    run_result_show(filename)

if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### DATA OPTION
    arg('--flag', type=bool, default=True, help='수정된 데이터 사용여부를 설정할 수 있습니다.')


    ############### BASIC OPTION
    arg('--data_path', type=str, default='/opt/ml/data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')

    arg('--context_data',type=str,default='baseline',help='context_data 전처리 파일을 선택할 수 있습니다.')
    ############### TRAINING OPTION
    arg('--batch_size', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--lr', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')
    arg('--early_stop', type=int, default=1, help='early stop을 조정할 수 있습니다.')


    ############### KFold Option
    arg('--kfold', type=int, default=1, help='KFold Option (0: False, 1:True) -> only StratifiedKFold is supported')
    arg('--kfold_n_splits', type=int, default=5, help='KFold the number of split')


    ############### GPU
    arg('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')


    ############### FM, FFM, NCF, WDN, DCN Common OPTION
    arg('--embed_dim', type=int, default=16, help='FM, FFM, NCF, WDN, DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--dropout', type=float, default=0.2, help='NCF, WDN, DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--mlp_dims', type=list, default=(16, 16), help='NCF, WDN, DCN에서 MLP Network의 차원을 조정할 수 있습니다.')


    ############### DCN
    arg('--num_layers', type=int, default=3, help='에서 Cross Network의 레이어 수를 조정할 수 있습니다.')


    ############### CNN_FM
    arg('--cnn_embed_dim', type=int, default=64, help='CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--cnn_latent_dim', type=int, default=12, help='CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')


    ############### DeepCoNN
    arg('--vector_create', type=bool, default=False, help='DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.')
    arg('--deepconn_embed_dim', type=int, default=32, help='DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--deepconn_latent_dim', type=int, default=10, help='DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')
    arg('--conv_1d_out_dim', type=int, default=50, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    arg('--kernel_size', type=int, default=3, help='DEEP_CONN에서 1D conv의 kernel 크기를 조정할 수 있습니다.')
    arg('--word_dim', type=int, default=768, help='DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다.')
    arg('--out_dim', type=int, default=32, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')


    args = parser.parse_args()
    main(args)
