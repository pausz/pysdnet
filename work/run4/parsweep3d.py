random.seed(42)
ngrid         = 128
vel, gsc, exc = logspace(0, 1.5, ngrid), logspace(-6., -1., ngrid), r_[0.95:1.01:ngrid*1j]
datasets      = map(data.dsi.load_dataset, [0])
models        = ['fhn_euler']
nic           = 16
dt            = 0.5
tf            = 300
ds            = 10
j             = 0
start_time    = time.time()
launch_count  = 0

