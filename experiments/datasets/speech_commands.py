import os
import pathlib
import urllib.request
import tarfile
import torch
import torchaudio

from . import common

import pdb
#here = pathlib.Path(__file__).resolve().parent
here = pathlib.Path("/dfs/scratch1/ksaab")

def download():
    
    base_base_loc = here / 'data'
    base_loc = base_base_loc / 'SpeechCommands'
    loc = base_loc / 'speech_commands.tar.gz'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_base_loc):
        os.mkdir(base_base_loc)
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    urllib.request.urlretrieve('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz', loc)
    with tarfile.open(loc, 'r') as f:
        f.extractall(base_loc)


def _process_data(intensity_data, dropped_rate=0, raw_data=False):
    base_loc = here / 'data' / 'SpeechCommands'
    X = torch.empty(34975, 16000, 1)
    y = torch.empty(34975, dtype=torch.long)

    batch_index = 0
    y_index = 0
    for foldername in ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'):
        loc = base_loc / foldername
        for filename in os.listdir(loc):
            # audio, _ = torchaudio.load_wav(loc / filename, channels_first=False,
            #                                normalization=False)  # for forward compatbility if they fix it
            audio, _ = torchaudio.load(
                    loc / filename, channels_first=False,
                )
            audio = audio / 2 ** 15  # Normalization argument doesn't seem to work so we do it manually.
            
            # A few samples are shorter than the full length; for simplicity we discard them.
            if len(audio) != 16000:
                continue

            X[batch_index] = audio
            y[batch_index] = y_index
            batch_index += 1
        y_index += 1
    assert batch_index == 34975, "batch_index is {}".format(batch_index)

    if not raw_data:
        X = torchaudio.transforms.MFCC(log_mels=True, n_mfcc=20,
                                    melkwargs=dict(n_fft=200, n_mels=64))(X.squeeze(-1)).transpose(1, 2).detach()
        # X is of shape (batch=34975, length=161, channels=20)

    ## KS ADD: removing p% of points
    if dropped_rate != 0:
        generator = torch.Generator().manual_seed(56789)
        X_removed = []
        for Xi in X:
            removed_points = (
                torch.randperm(X.shape[-1], generator=generator)[
                : int(X.shape[-1] * float(dropped_rate) / 100.0)
                ]
                    .sort()
                    .values
            )
            Xi_removed = Xi.clone()
            Xi_removed[:, removed_points] = float("nan")
            X_removed.append(Xi_removed)
        X = torch.stack(X_removed, dim=0)


    times = torch.linspace(0, X.size(1) - 1, X.size(1))
    final_index = torch.tensor(X.size(1) - 1).repeat(X.size(0))

    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, _) = common.preprocess_data(times, X, y, final_index, append_times=True,
                                                   append_intensity=intensity_data)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index)


def get_data(intensity_data, batch_size, dropped_rate=0, raw_data=False):
    #base_base_loc = here / 'processed_data'
    in1 = '_intensity' if intensity_data else ''
    in2 = '_droprate_' + str(dropped_rate) if dropped_rate > 0 else ''
    in3 = '_raw' if raw_data else ''
    appnd = 'processed_data'+in1+in2+in3
    base_base_loc = here / appnd
    loc = base_base_loc / ('speech_commands_with_mels' + ('_intensity' if intensity_data else ''))

    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
    else:
        #download()
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index) = _process_data(intensity_data, dropped_rate, raw_data)
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        common.save_data(loc, times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index)

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, 'cpu',
                                                                                batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader
