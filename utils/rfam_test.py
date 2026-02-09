def generate_samples(autoencoder, dataloader, num_samples,vocab,vocab_str):
    autoencoder.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        # Get a batch of sequences from the dataloader
        for sequences,_,structures in dataloader:
            # Move sequences to GPU
            sequences = sequences.to(device)
            structures = structures.to(device)
            mask = create_mask(sequences, vocab['P']).to(device)

            # Get latent representations
            latent = autoencoder.encoder(sequences,structures,mask)

            # Generate new sequences from the latent space
            generated_sequences,generated_str= autoencoder.decoder(latent)

            # Convert generated sequences to class indices
            generated_indices = torch.argmax(generated_sequences, dim=-1)  # Shape: [batch_size, seq_len]
            generated_indices_str = torch.argmax(generated_str, dim=-1)
            re_mask=1-mask
            # Break after generating the required number of samples
            for i in range(num_samples):
                sequence = indices_to_sequence(generated_indices[i], vocab)
                sequence=modify_with_mask(sequence,re_mask[i])
                # print(1 - mask)
                structure =indices_to_structure(generated_indices_str[i], vocab_str)
                structure=modify_with_mask(structure, re_mask[i])
                print(f'Generated Sequence {i + 1}: {sequence}')
                print(f'Generated Structure {i + 1}: {structure}')

            break  # Exit the loop after processing one batch