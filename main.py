import torch
from dataset import *
import argparse
import os
from model_clam import MultimodalModel, CLAM_mre, CLAM_endo
from torch.utils.data import DataLoader
from utils.topk.svm import SmoothTop1SVM
from utils.utils import set_seeds, create_datasets_for_fold, train, validate, test




def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate MWMIL model for Pediatric Crohn's Disease")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model (default: cuda:0)')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience level')
    parser.add_argument('--folder_name', type=str, default="name", help='Folder name for saving results')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    results_dir = f'./result_main/{args.folder_name}'
    device=args.device

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for fold in range(10):  # 10 fold cross validation
        split_file = f'splits_{fold}.csv'
        train_dataset, val_dataset, test_dataset = create_datasets_for_fold(
            split_file,
            pt_directory='pt_files_directory',
            mre_directory='mre_directory',
            label_csv='lable_directory',
            tabular_csv='tabular_directory',
            mre_endo_csv='mre_endo_directory'
        )

        train_loader = get_split_loader(train_dataset, training=True, weighted=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        cross_loss_fn = torch.nn.CrossEntropyLoss().to(device)
        instance_loss_fn = SmoothTop1SVM(n_classes=2).to(device)
        endo_model = CLAM_endo()
        mre_model = CLAM_mre()
        model = MultimodalModel(endo_model=endo_model, mre_model=mre_model, instance_loss_fn=instance_loss_fn)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = cross_loss_fn

        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = 30  # your patience level

        for epoch in range(args.num_epochs):
            print(f'epoch:{epoch}')
            train(model, train_loader, optimizer, criterion, device)
            val_loss = validate(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_path = os.path.join(results_dir, f'best_model_fold_{fold}.pt')
                torch.save(model.state_dict(), best_model_path)
                print(f'Model saved: Improved validation loss to {best_val_loss}')
            else:
                epochs_no_improve += 1
                print(f'No improvement in validation loss for {epochs_no_improve} epochs')

            if epochs_no_improve == patience:
                print('Early stopping triggered')
                break

        model.load_state_dict(torch.load(best_model_path))
        test_accuracy = test(model, test_loader, fold, split_file, results_dir, device)

        print(f"Fold {fold}: Test Accuracy = {test_accuracy}")


if __name__ == "__main__":
    main()
