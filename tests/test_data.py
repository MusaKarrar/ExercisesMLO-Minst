from tests import _PATH_DATA
from project_name.data.make_dataset import mnist
def test_data():
 
    train_dataset, test_dataset = mnist()
    
    
    
    assert len(test_dataset) == 5000
    #assert len(test_dataset) == 4000
    #and N_test for test
    #assert that each datapoint has shape [1,28,28] or [784] depending on how you choose to format
    #assert that all labels are represented