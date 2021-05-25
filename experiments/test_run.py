import speech_commands

#speech_commands.main(model_name="ncde",hidden_channels=128,hidden_hidden_channels=64,num_hidden_layers=4)
speech_commands.run_all(device='cuda', raw_data=True)