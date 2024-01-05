import torch


def mnist():
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [ ], [ ]
    for i in range(5):
        train_data.append(torch.load(f"data/raw/train_images_{i}.pt"))
        train_labels.append(torch.load(f"data/raw/train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_data = torch.load("data/raw/test_images.pt")
    test_labels = torch.load("data/raw/test_target.pt")

    torch.save(train_data, "data/processed/train_data.pt")
    torch.save(train_labels, "data/processed/trainlabels.pt")
    torch.save(test_data, "data/processed/testdata.pt")
    torch.save(test_labels, "data/processed/testlabel.pt")

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    return (
        torch.utils.data.TensorDataset(train_data, train_labels), 
        torch.utils.data.TensorDataset(test_data, test_labels)
    )
if __name__ == "__main__":
     mnist()