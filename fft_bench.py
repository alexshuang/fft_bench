import argparse
import json
import os
import docopt
import torch
import contexttimer


def bench(batch_size:int, d:int, hw:int, num_iter:int):
    if not torch.cuda.is_available():
        print("GPU is not available")
        return

    device = torch.device('cuda:0')

    torch.set_grad_enabled(False)

    # BxDxHxWx2
    inp = torch.randn(batch_size, d, hw, hw, 2, device=device)

    # warmup
    outp = torch.fft(inp, 3)
    inp_ = torch.ifft(outp, 3)

    # fft
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with contexttimer.Timer() as t:
        for it in range(num_iter):
            outp = torch.fft(inp, 3)
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / 1e3
    tps = num_iter / elapsed
    fft_time_consume = elapsed

    del outp, inp

    outp = torch.randn(batch_size, d, hw, hw, 2, device=device)

    # ifft
    start.record()
    with contexttimer.Timer() as t:
        for it in range(num_iter):
            inp_ = torch.ifft(outp, 3)
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / 1e3
    itps = num_iter / elapsed
    ifft_time_consume = elapsed

    print(
        json.dumps({
            "TPS": tps,
            "fft_elapsed": fft_time_consume,
            "ITPS": itps,
            "ifft_elapsed": ifft_time_consume,
            "n": num_iter,
            "batch_size": batch_size,
            "D_size": d,
            "HW_size": hw,
        }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "Replacement kernel generator"
    parser.add_argument("-b", "--batch_size", type=int, help='')
    parser.add_argument("-d", "--d_size", type=int, help='')
    parser.add_argument("-n", "--num_iter", type=int, help='')
    parser.add_argument("-hw", "--hw_size", type=int, help='')
    args = parser.parse_args()

    bench(args.batch_size, args.d_size, args.hw_size, args.num_iter)

#    kwargs = {
#        'batch_size': int(args.batch_size),
#        'n': int(args.num_iter),
#        'd': int(args.d_size),
#        'hw': int(args.hw_size),
#    }
#    bench(**kwargs)

