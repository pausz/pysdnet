    random.seed(42)
    ngrid         = 64
    vel, gsc, exc = logspace(-0.3, 2, ngrid), logspace(-6., -0.5, ngrid), r_[0.8:1.2:ngrid*1j]
    datasets      = map(data.dsi.load_dataset, [0])
    models        = ['fhn_euler']
    nic           = 32
    dt            = 0.5
    tf            = 1000
    ds            = 5
    j             = 0
    start_time    = time.time()
    launch_count  = 0


