#!/usr/bin/python

import numpy as np

np.random.seed(97)

for bs in [1, 4, 16, 64]:
    for sl in [16, 64, 256, 1024]:
        fileName = "./data/encoder-%d-%d.npz" % (bs, sl)
        data = {}
        data['speech'] = np.random.rand(bs * sl * 80).astype(np.float32).reshape(bs, sl, 80) * 2 - 1
        data['speech_lengths'] = np.random.randint(1, sl, [bs]).astype(np.int32)
        data['encoder_out'] = np.random.rand(bs * (sl // 4 - 1) * 256).astype(np.float32).reshape(bs, (sl // 4 - 1), 256) * 2 - 1
        data['encoder_out_lens'] = np.random.randint(1, sl, [bs, 1, (sl // 4 - 1)]).astype(bool).reshape(bs, 1, (sl // 4 - 1))
        np.savez(fileName, **data)

for bs in [1, 4, 16, 64]:
    for sl in [16, 64, 256, 1024]:
        fileName = "./data/decoder-%d-%d.npz" % (bs, sl)
        data = {}
        data['encoder_out'] = np.random.rand(bs * sl * 256).astype(np.float32).reshape(bs, sl, 256) * 2 - 1
        data['encoder_out_lens'] = np.random.randint(1, sl, [bs]).astype(np.int32)
        #data['hyps_pad_sos_eos'] = np.random.randint(0,10,[bs,10,sl]).astype(np.int32)
        data['hyps_pad_sos_eos'] = np.random.randint(0, 10, [bs, 10, 64]).astype(np.int32)
        data['hyps_lens_sos'] = np.random.randint(0, 10, [bs, 10]).astype(np.int32)
        data['ctc_score'] = np.random.rand(bs * 10).astype(np.float32).reshape(bs, 10) * 2 - 1
        #data['decoder_out'] = np.random.rand(bs*10*(sl-1)*4233).astype(np.float32).reshape(bs,10,sl-1,4233) * 2 - 1
        data['decoder_out'] = np.random.rand(bs * 10 * (64 - 1) * 4233).astype(np.float32).reshape(bs, 10, (64 - 1), 4233) * 2 - 1
        data['best_index'] = np.random.randint(0, 10, [bs]).astype(np.int32)

        np.savez(fileName, **data)

print("Finish!")
