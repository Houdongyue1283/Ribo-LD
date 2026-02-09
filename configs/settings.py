# mode = "LINE1_evaluate"
mode = "train" 
# mode = "Rfam_evaluate" 

if_att =  True

seq_w = 0.5

if_process_dataset = 0


vocab = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'P': 4}  

if_hidden_str= False

if if_hidden_str==True:
    structure_vocab = {'0': 0, '1': 1}
else:
    structure_vocab = {'(': 0, ')': 1, '.': 2, 'P': 3}  # Vocabulary for secondary structures

max_len = 200

latent_dim = 256

num_epochs = 200

batch_size = 32

learning_rate = 1e-4

loss_tp="CE"